from ultralytics import YOLO
import numpy as np
import os
import os.path as osp
import time
import cv2
import torch
import torchvision
from sort.sort import *

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

        self.duration = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            self.duration = self.average_time
        else:
            self.duration = self.diff
        return self.duration

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.duration = 0.

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    new_h, new_w = new_shape

    # scale factor
    scale = min(new_w / w, new_h / h)
    resized_w, resized_h = int(w * scale), int(h * scale)

    # resize
    img_resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    # padding 
    pad_w = new_w - resized_w
    pad_h = new_h - resized_h
    pad_left   = pad_w // 2
    pad_right  = pad_w - pad_left
    pad_top    = pad_h // 2
    pad_bottom = pad_h - pad_top

    # padding
    img_padded = cv2.copyMakeBorder(
        img_resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return img_padded, scale, (pad_left, pad_top)

def transform_cv(image):    
    # 0) BGR -> RGB (필요 시 주석 해제)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 1) Resize (640x640)
    # image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)

    # 2) ToTensor (HWC -> CHW, 0~1 float)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # (H,W,C) -> (C,H,W)

    # 3) Add batch dimension
    image = np.expand_dims(image, axis=0)  # (1,C,H,W)

    # Return as NumPy array (C-order)   
    return np.array(image, dtype=np.float32, order="C")

def scale_boxes_back(boxes, scale, pad, orig_shape):
    """
    boxes: [N,4], (x1, y1, x2, y2) in resized/letterbox coords
    scale: float, resizing
    pad: (pad_left, pad_top)
    orig_shape: (H_orig, W_orig)
    """
    pad_left, pad_top = pad
    H_orig, W_orig = orig_shape

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W_orig)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H_orig)

    return boxes

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im

class YOLO11(torch.nn.Module):
    def __init__(self, checkpoint_path, class_count=80) -> None:
        super().__init__()

        model = YOLO(checkpoint_path)
        self.model = model.model
        self.class_count = class_count

    def forward(self, x):
        """https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L216"""
        pred: torch.Tensor = self.model(x)[0]  # [N, 84, 8400]
        pred1 = pred.permute(0, 2, 1) # [N, 84, 8400] -> [N, 8400, 84] 
        boxes, scores = pred1.split([4, self.class_count], dim=-1) # [N, 8400, 84] -> [N, 8400, 4], [N, 8400, 80]  
        boxes = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        
        return boxes, scores

def main():
    video_path = f"{CUR_DIR}/../../data/video/palace.mp4"
    filename = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    current_time = time.localtime()
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(CUR_DIR, "results", timestamp)
    os.makedirs(save_folder, exist_ok=True)
    save_path = osp.join(save_folder, f"{filename}.mp4")

    print(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))

    timer = Timer()
    frame_id = 0

    # Load the YOLO11 model
    model_name = "yolo11n"
    model = YOLO11(model_name)

    #create instance of the SORT tracker
    mot_tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3) 
    while True:
        if frame_id % 20 == 0:
            print('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            timer.tic()
            # Detect objects
            raw_img = frame.copy()
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            padded_image, scale, pad = letterbox(rgb_image)
            preproc_image = transform_cv(padded_image)  # Preprocess image
            tensor_image = torch.from_numpy(preproc_image)
            with torch.no_grad():
                output = model(tensor_image)  # predict on an image

            pred_boxes, scores = output
            pred_scores, pred_labels = torch.max(scores, dim=-1)

            pred_label = pred_labels[0]
            pred_box = pred_boxes[0]
            pred_score = pred_scores[0]

            # NMS 
            iou_threshold=0.5
            keep = torchvision.ops.batched_nms(pred_box, pred_score, pred_label, iou_threshold)
            keep_topk = 300
            keep = keep[: keep_topk]

            labels = pred_label[keep]
            boxes = pred_box[keep]
            scores = pred_score[keep]

            thrh=0.45
            scr = scores
            lab = labels[scr > thrh]
      
            if len(lab) == 0:  
                trackers = mot_tracker.update()
            else:
                box = boxes[scr > thrh]
                scrs = scr[scr > thrh]
                box = scale_boxes_back(box, scale, pad, (height, width))
                dets = torch.cat((box, scrs.unsqueeze(1)), dim=1) # [x1,y1,x2,y2,score]
                trackers = mot_tracker.update(dets)

            online_tlwhs = []
            online_ids = []
            for t in trackers:
                tlwh = [t[0], t[1], t[2]-t[0], t[3]-t[1]]
                tid = t[4]
                if tlwh[2] * tlwh[3] > 10:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)

            timer.toc()
            online_im = plot_tracking(
                raw_img, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
            )
            
            # Display the annotated frame
            cv2.imshow("SORT with YOLO11", online_im)

            vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1



if __name__ == '__main__':
    main()