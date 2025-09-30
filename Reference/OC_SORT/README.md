# OC_SORT


1. set up a virtual environment.
    ```
    git clone https://github.com/noahcao/OC_SORT.git
    cd OC_SORT
    conda create -n ocsort -y python=3.11
    conda activate ocsort
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

    # comment line 22 in requirements.txt (# onnx-simplifier==0.3.5)
    pip install -r requirements.txt
    pip install onnxsim
    pip install onnxruntime
    python setup.py develop

    pip install cython
    pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    pip install cython_bbox
    pip install pandas
    pip install xmltodict
    pip install gdown
    pip install loguru
    pip install opencv-python
    pip install thop
    pip install tabulate
    pip install filterpy
    ```

2. download pretrained checkpoints.
    ```
    mkdir -p pretrained
    gdown --fuzzy https://drive.google.com/file/d/1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj/view?usp=sharing -O pretrained/
    ```

3. run demo from original repository
    ```
    python tools/demo_track.py --demo_type video -f exps/example/mot/yolox_s_mix_det.py -c pretrained/bytetrack_s_mot17.pth.tar --path ../../../data/video/palace.mp4 --fp16 --fuse --save_result --out_path demo_out.mp4
    ```

4. check simple run
    ```
    cd ..
    python ocsort_track.py
    ```

---

- [Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking](https://arxiv.org/pdf/2203.14360)
- [OC_SORT official GitHub](https://github.com/noahcao/OC_SORT)
---