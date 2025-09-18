# Multi_Object_Tracking


1. set up a virtual environment.
    ```
    https://github.com/FoundationVision/ByteTrack.git
    cd ByteTrack
    conda create -n btrack -y python=3.11
    conda activate btrack
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

    # comment line 23 in requirements.txt (# onnx-simplifier==0.4.10)
    pip install -r requirements.txt
    python setup.py develop

    pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    pip install cython_bbox
    pip install gdown
    ```

2. download pretrained checkpoints.
    ```
    mkdir -p pretrained
    gdown --fuzzy https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view?usp=sharing -O pretrained/
    gdown --fuzzy https://drive.google.com/file/d/1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj/view?usp=sharing -O pretrained/
    ```
3. run demo from original repository
    ```
    python3 tools/demo_track.py video -f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result
    ```

4. check simple run
    ```
    cd ..
    python byte_track.py
    ```

---
- Change detector

1. download pretrained checkpoints from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
    ```
    wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth -P ByteTrack/pretrained
    ```
2. check simple run
    ```
    python yolox_x_track.py
    ```
