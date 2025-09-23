# YOLOX: Exceeding YOLO Series in 2021

## How to Run

1. set up a virtual environment.
    ```
    git clone https://github.com/Megvii-BaseDetection/YOLOX.git
    cd YOLOX
    conda create -n yolox -y python=3.11
    conda activate yolox
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

    # comment line 18 in requirements.txt (# onnx-simplifier==0.4.10)
    pip install -v -e .
    pip install opencv-python
    pip install onnx
    pip install onnxscript
    pip install onnxsim
    pip install onnx_graphsurgeon
    ```

2. download pretrained.
    ```
    mkdir -p pretrained
    wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth -P pretrained
    ```
    
3. . generate onnx file
    ```
    cd ..
    python onnx_export.py
    ```

- [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/pdf/2107.08430)
- [YOLOX official GitHub](https://github.com/Megvii-BaseDetection/YOLOX)

