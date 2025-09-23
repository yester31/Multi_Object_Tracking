# YOLOX: Exceeding YOLO Series in 2021

## How to Run

1. set up a virtual environment.
    ```
    git clone https://github.com/FoundationVision/ByteTrack.git
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
    pip install onnx
    pip install onnxscript
    pip install onnxsim
    pip install onnx_graphsurgeon
    ```

2. download pretrained checkpoints.
    ```
    mkdir -p pretrained
    gdown --fuzzy https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view?usp=sharing -O pretrained/
    ```
    
3. . generate onnx file
    ```
    cd ..
    python onnx_export.py
    ```

- [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/pdf/2110.06864)
- [ByteTrack official GitHub](https://github.com/FoundationVision/ByteTrack)

