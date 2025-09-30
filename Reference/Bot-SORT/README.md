# BoT-SORT


1. set up a virtual environment.
    ```
    git clone https://github.com/NirAharon/BoT-SORT.git
    cd BoT-SORT

    conda create -n botsort -y python=3.9
    conda activate botsort
    
    pip install --force-reinstall "numpy==1.23.5"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
    pip install gdown
    pip install opencv-python
    pip install onnx
    pip install onnxscript
    pip install onnxsim
    pip install loguru
    pip install thop
    pip install tabulate
    pip install scikit-image
    pip install scikit-learn
    pip install lap
    pip install pyyaml
    pip install yacs
    pip install termcolor
    pip install tensorboard
    pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    pip install cython
    pip install cython_bbox
    pip install faiss-cpu
    pip install faiss-gpu
    ```

2. download pretrained checkpoints.
    ```
    mkdir -p pretrained
    gdown --fuzzy https://drive.google.com/file/d/1QZFWpoa80rqo7O-HXmlss8J8CnS7IUsN/view?usp=sharing -O pretrained/
    ```

3. run demo from original repository
    ```
    python tools/mc_demo.py video --path ../../data/video/palace.mp4 -f yolox/exps/example/mot/yolox_s_mix_det.py -c pretrained/bytetrack_s_mot17.pth.tar --with-reid --fuse-score --fp16 --fuse --save_result
    ```

4. check simple run
    ```
    cd ..
    python bot_sort.py
    ```
    
- [BoT-SORT: Robust Associations Multi-Pedestrian Tracking](https://arxiv.org/pdf/2206.14651)
- [BoT-SORT official GitHub](https://github.com/NirAharon/BoT-SORT)
---

