# Self-Evolving Depth-Supervised 3D Gaussian Splatting from Rendered Stereo Pairs (BMVC 2024)

### [Project Page](https://kuis-ai.github.io/StereoGS/) | [Paper](https://arxiv.org/abs/2409.07456)

This repository contains the code for our work "**Self-Evolving Depth-Supervised 3D Gaussian Splatting from Rendered Stereo Pairs**", [BMVC 2024](https://bmvc2024.org/)

by [Sadra Safadoust](https://sadrasafa.github.io/), [Fabio Tosi](https://fabiotosi92.github.io/), [Fatma Güney](https://mysite.ku.edu.tr/fguney/), and [Matteo Poggi](https://mattpoggi.github.io/)

### [Project Page](https://kuis-ai.github.io/multi-object-segmentation/) | [Paper](https://arxiv.org/abs/2307.08027)


## :bookmark_tabs: Table of Contents

1. [Installation](#gear-installation)
2. [Datasets](#file_cabinet-datasets)
3. [Training](#watermelon-training)
4. [Citation](#fountain_pen-citation)

## :gear: Installation

Create a conda environment and install the requirements:
```
conda create -n StereoGS python=3.11
conda activate StereoGS
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install plyfile tqdm
```

Clone the repositroy and its submodules, and install the submodules (Note that we use a different rasterize than the original [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)\):
```
git clone https://github.com/sadrasafa/StereoGS.git --recursive
pip install submodules/depth-diff-gaussian-rasterization
pip install submodules/simple-knn
```

Clone [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) into `utils\`. install its requirements, download its checkpoints, and compile the CUDA implementation of correlation sampler:
```
cd utils
git clone https://github.com/princeton-vl/RAFT-Stereo.git
pip install matplotlib tensorboard scipy opencv-python opt_einsum imageio scikit-image
cd RAFT-Stereo/sampler && python setup.py install && cd ../
bash download_models.sh
cd ..
```


## :file_cabinet: Datasets
We experiment on three datasets: ETH3D, ScanNet++, and BlendedMVS. For each of the datasets, you should store them in the following structure after processing them (described below):

```
├── [DATASET-NAME]
    ├── [SCENE-NAME]
        ├── val_cams.json
        ├── images
        ├── depths
        ├── sparse
            ├── 0
                ├── cameras.txt
                ├── images.txt
                ├── points3D.txt
        
```            
You can ignore `val_cams.json` if you don't want to have the same train/val split as ours. You can download them for each scene from [here](here).

### 1. ETH3D

Download the High-res multi-view dataset from [here](https://www.eth3d.net/datasets).
We only need the undistorted jpg images and the undistorted depths. Note that only ground-truth depth that match the distorted images are provided, therefore they need to be undistorted given the camera parameters.

### 2. ScanNet++

Download the dataset from [here](https://kaldir.vc.in.tum.de/scannetpp/). We use the DSLR data for the `8b5caf3398` and `b20a261fdf` scenes. Follow the instructions at [Official ScanNet++ Toolkit](https://github.com/scannetpp/scannetpp) to render depth and then undistort the images and depths. Note that the provided undistortion script only undistorts the images, however it can be easily extended to undistort depths too (e.g., check [this](https://github.com/scannetpp/scannetpp/issues/65#issuecomment-1939346286)). Also, it saves the camera intrinsics for the undistorted pinhle camera in the nerfstudio's json format, so make sure to update the camera parameters in the colmap format (`cameras.txt`) accordingly as well.

### 3. BlendedMVS

Download the low-res BlendedMVS dataset from [here](https://github.com/YoYo000/BlendedMVS). We use the following scenes:
```
5b6e716d67b396324c2d77cb
5b6eff8b67b396324c5b2672
5bf18642c50e6f7f8bdbd492
5bff3c5cfe0ea555e6bcbf3a
```
We remove `*_masked.jpg` files.\
The dataset provides the camera parameters but does not provide the COLMAP SfM points. You can follow the instructions [here](https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses) to obtain the sparse COLMAP model using the given poses.

## :watermelon: Training

You can train and evaluate the method on each of the datasets using the [scripts](scripts) provided.
Set the appropriate dataset path and connection port and then run the scripts:

```
bash scripts/run_ETH3D.sh
bash scripts/run_scannetpp.sh
bash scripts/run_blendedMVS.sh
```

## :fountain_pen: Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{safadoust2024BMVC,
  title={Self-Evolving Depth-Supervised 3D Gaussian Splatting from Rendered Stereo Pairs},
  author={Safadoust, Sadra and Tosi, Fabio and G{\"u}ney, Fatma and Poggi, Matteo},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2024}
}
```