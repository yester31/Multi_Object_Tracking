# SORT


1. set up a virtual environment.
    ```
    git clone https://github.com/abewley/sort.git
    cd sort
    conda create -n ocsort -y python=3.11
    conda activate ocsort
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
    pip install filterpy scikit-image lap matplotlib
    ```

2. check simple run
    ```
    cd ..
    python sort_track.py
    ```
---

- [SIMPLE ONLINE AND REALTIME TRACKING](https://arxiv.org/pdf/1602.00763)
- [SORT official GitHub](https://github.com/abewley/sort)
---