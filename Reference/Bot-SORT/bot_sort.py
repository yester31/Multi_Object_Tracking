import sys
import os
import os.path as osp
import time
import cv2
import torch

sys.path.insert(1, os.path.join(sys.path[0], "BoT-SORT"))
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking

from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class fake_args:
    def __init__(self):
        # Add required attributes
        self.track_thresh = 0.5   # Default threshold for ByteTrack
        self.mot20 = False
        self.aspect_ratio_thresh=1.6
        self.save_result=True

        self.device="gpu"
        self.track_high_thresh=0.6
        self.track_low_thresh=0.1
        self.new_track_thresh=0.7
        self.track_buffer=30
        self.match_thresh=0.8
        self.min_box_area=10
        self.fuse_score=False

        self.cmc_method="sparseOptFlow"
        self.name=None
        self.ablation=False

        self.proximity_thresh = 0.5
        self.appearance_thresh = 0.25

        self.with_reid = False
        self.fast_reid_config = f"{CUR_DIR}/BoT-SORT/fast_reid/configs/MOT17/sbs_S50.yml"
        self.fast_reid_weights = f"{CUR_DIR}/BoT-SORT/pretrained/mot17_sbs_S50.pth"

def main():

    model_name = None
    exp_file = f"{CUR_DIR}/BoT-SORT/yolox/exps/example/mot/yolox_s_mix_det.py"
    exp = get_exp(exp_file, model_name)

    vis_folder = osp.join(CUR_DIR, "results", exp.exp_name)
    os.makedirs(vis_folder, exist_ok=True)

    exp.test_conf = 0.001
    exp.nmsthre = 0.7
    exp.test_size = (608, 1088)
    model = exp.get_model().to(DEVICE)
    print("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    ckpt_file = f"{CUR_DIR}/BoT-SORT/pretrained/bytetrack_s_mot17.pth.tar"
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"]) # load the model state dict
    print("loaded checkpoint done.")

    model = fuse_model(model)
    model = model.half()  # to FP16

    num_classes = 1
    confthre = 0.001
    nmsthre = 0.7 

    video_path = f"{CUR_DIR}/../../data/video/palace.mp4"
    filename = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    current_time = time.localtime()
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    save_path = osp.join(save_folder, f"{filename}.mp4")

    print(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))

    args = fake_args()
    args.with_reid = True
    tracker = BoTSORT(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    scale = min(exp.test_size[0] / float(height), exp.test_size[1] / float(width))
    while True:
        if frame_id % 20 == 0:
            print('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            # Detect objects
            raw_img = frame.copy()
            img, ratio = preproc(frame, exp.test_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            img = torch.from_numpy(img).unsqueeze(0).float().to(DEVICE).half()  # to FP16
            with torch.no_grad():
                timer.tic()
                outputs = model(img)
                outputs = postprocess(outputs, num_classes, confthre, nmsthre)
            detections = []
            if outputs[0] is not None:
                outputs = outputs[0].cpu().numpy()
                detections = outputs[:, :7]
                detections[:, :4] /= scale

            # Run tracker
            online_targets = tracker.update(detections, raw_img)

            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                if tlwh[2] * tlwh[3] > args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                raw_img, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
            )
            # else:
            #     timer.toc()
            #     online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        print(f"save results to {res_file}")


if __name__ == "__main__":
    main()
