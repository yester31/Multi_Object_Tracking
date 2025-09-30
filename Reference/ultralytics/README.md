# ultralytics Tracking

## How to Run

1. set target model (yolo11n)
    ```
    git clone https://github.com/ultralytics/ultralytics
    cd ultralytics
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
    pip install ultralytics
    ```

2. generate .onnx from timm model
    ```
    cd ..
    python onnx_export.py
    ```