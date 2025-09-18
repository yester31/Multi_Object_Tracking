import cv2
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], "ultralytics"))
from ultralytics import YOLO
import torch

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    # choose a tracker type
    tracker_type ="botsort" # botsort or bytetrack

    # Load the YOLO11 model
    model_name = "yolo11n"
    checkpoint_path = f'{CUR_DIR}/ultralytics/checkpoints/{model_name}.pt'
    model = YOLO(checkpoint_path)

    # Open the video file
    video_path = f"{CUR_DIR}/../../data/video/video1.mp4" # video1 or palace
    filename = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{save_dir_path}/{filename}_{model_name}_{tracker_type}.mp4', fourcc, 30.0, (w, h))

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, conf=0.5, iou=0.5, tracker=f"{tracker_type}.yaml") 

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLO11 Tracking", annotated_frame)

            resized = cv2.resize(annotated_frame, (w, h))  # (width, height)
            out.write(resized)  # write video frame

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()