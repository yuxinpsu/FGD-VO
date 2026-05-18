# FGD-VO: An End-to-End Inter-Frame Correspondence Modeling Network for Monocular Visual Odometry
Official repository of the paper: Enhancing Autonomous Vehicle Visual Odometry with Reliable Inter-Frame Correspondence Modeling

## Abstract
*Visual odometry (VO) provides continuous ego-motion estimates for autonomous vehicles (AVs) and is especially critical when Global Navigation Satellite System signals are degraded, such as in urban canyons, tunnels, and overpasses. Recent learning-based monocular VO methods have shown promising accuracy, yet their performance remains constrained by the quality of inter-frame correspondences—the pixel-level associations between consecutive images from which camera motion is inferred. Two aspects of this problem are insufficiently addressed. First, existing methods often lack explicit inter-frame feature alignment and rely on implicit associations, which break down under the spatially non-uniform displacement typical of driving: at highway speed, nearby lane markings shift substantially while distant overpasses barely move, and turning maneuvers at intersections introduce large rotational displacement. Second, correspondences are routinely corrupted by road-specific conditions—independent motion from surrounding vehicles and pedestrians, abrupt lighting transitions at tunnel entrances, and specular reflections from wet road surfaces—yet are not effectively suppressed before pose estimation. To address these issues, we propose FGD-VO, a reliable inter-frame correspondence modeling framework for monocular VO. A Flow-Guided Deformable alignment module uses dense optical flow to provide coarse correspondence priors and refines them with learnable local offsets, enabling adaptive feature alignment under spatially varying displacement. A Dual-Path Reliability Masking strategy combines an explicit flow-consistency cue with a learnable attention mask to suppress unreliable correspondences while emphasizing informative regions. Experiments on the KITTI odometry benchmark show that FGD-VO achieves lower mean translational and rotational errors than representative learning-based monocular VO baselines under the evaluated protocols, demonstrating the benefit of explicit correspondence alignment and reliability-aware masking for driving-scene visual odometry.*

<img src="doc/FGDVO.png" width=1000>

## 1. Dataset
Download the [KITTI odometry dataset (color).](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
The data structure should be as follows:
```
|---data_odometry_color
    |---dataset
        |---sequences
            |---00
                |---image_2
                    |---000000.png
                    |---000001.png
                    |---...
                |---image_3
                    |...
                |---calib.txt
                |---times.txt          
            |---01
            |---...
```

## 2. Setup
- Create a virtual environment using Anaconda and activate it:
```
conda create -n fgdvo python==3.8.0
conda activate fgdvo
```

- Install PyTorch and CUDA dependencies:
```
conda install pytorch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

- Install [RAFT](https://github.com/princeton-vl/raft):
```
git clone https://github.com/yuxinpsu/FGD-VO.git
cd FGD-VO/third party/
git clone https://github.com/princeton-vl/RAFT.git
cd RAFT
./download_models.sh
```

## 3. Train
Please configure the training sequences before running
```
Python main.py --data_root /PATH/sequences
```

## 4. Inference
Please configure the checkpoint and dataset paths before running
```
Python inference.py
```

## 5. Evaluation 
Please refer to the [PWCLONet](https://github.com/IRMVLab/PWCLONet/tree/main) repository for more details about the evaluation metrics and how to run the evaluation toolbox.

## 6. Results and pre-trained models
Here you find the results and the pre-trained models:
(1) For the model trianed on sequeces 0, 1, 2, 8, and 9: 
<img src="doc/tab1.png" width=1000>
