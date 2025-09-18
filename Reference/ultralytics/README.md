# Multi_Object_Tracking


camid=0, 
save_result=True, 
exp_file='/home/workspace/Multi_Object_Tracking/reference/ByteTrack/tools/../exps/example/mot/yolox_x_mix_det.py', 
ckpt='/home/workspace/Multi_Object_Tracking/reference/ByteTrack/tools/../pretrained/bytetrack_x_mot17.pth.tar', 
device=device(type='cuda'), 
conf=None, 
nms=None, 
tsize=None, 
fps=30, 
fp16=True, 
fuse=True, 
trt=False, 
track_thresh=0.5, 
track_buffer=30, 
match_thresh=0.8, 
aspect_ratio_thresh=1.6, 
min_box_area=10, 
mot20=False)


conda create -n btrack -y python=3.11
conda activate btrack
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# comment line 23 in requirements.txt (# onnx-simplifier==0.4.10)
pip install -r requirements.txt
python setup.py develop

pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install cython_bbox
pip install gdown


mkdir -p pretrained
gdown --fuzzy https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view?usp=sharing -O pretrained/
gdown --fuzzy https://drive.google.com/file/d/1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj/view?usp=sharing -O pretrained/


python3 tools/demo_track.py video -f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result

