# GRAE-3DMOT: Geometry Relation-Aware Encoder for Online 3D Multi-Object Tracking    


# Introduction
This repo is the official Code ofGRAE-3DMOT: Geometry Relation-Aware Encoder for Online 3D Multi-Object Tracking (CVPR 2025). This is a beta version, so bugs may exist. We are currently working on code modifications.

# Abstract
Recently, 3D multi-object tracking (MOT) has widely adopted the standard tracking-by-detection paradigm, which solves the association problem between detections and tracks. Many tracking-by-detection approaches establish constrained relationships between detections and tracks using a distance threshold to reduce confusion during association. However, this approach does not effectively and comprehensively utilize the information regarding objects due to the constraints of the distance threshold. In this paper, we propose GRAE-3DMOT, Geometry Relation-Aware Encoder 3D Multi-Object Tracking, which contains a geometric relation-aware encoder to produce informative features for association. The geometric relation-aware encoder consists of three components: a spatial relation-aware encoder, a spatiotemporal relation-aware encoder, and a distance-aware feature fusion layer. The spatial relation-aware encoder effectively aggregates detection features by comprehensively exploiting as many detections as possible. The spatiotemporal relation-aware encoder provides spatiotemporal relation-aware features by combing spatial and temporal relation features, where the spatiotemporal relation-aware features are transformed into association scores for MOT. The distance-aware feature fusion layer is integrated into both encoders to enhance the relation features of physically proximate objects. Experimental results demonstrate that the proposed GRAE-3DMOT outperforms the state-of-the-art on the nuScenes. Our approach achieves 73.7\% and 70.2\% AMOTA on the nuScenes validation and test sets using CenterPoint detections.

<p align="center"> <img src='docs/overview.png', height="350px"> </p>   

# Setup environment

This fork is **self-contained** and managed with [uv](https://docs.astral.sh/uv/).
The two external dependencies the original README asked you to clone and build —
CenterPoint's `iou3d_nms` CUDA op and SimpleTrack — have been **re-implemented in
pure Python / PyTorch inside this repo** (see [`ops/`](ops/)), so there are **no
git submodules, no `pip install -e` of external projects, and no hand-built CUDA
extensions** to maintain.

The default environment targets **PyTorch built for CUDA 12.8**, which is required
for recent GPUs (e.g. NVIDIA Blackwell / `sm_120`). The `cu128` wheels are resolved
automatically from the dedicated PyTorch index configured in `pyproject.toml`.

## uv setup (recommended)

```shell
# install uv if needed: https://docs.astral.sh/uv/getting-started/installation/
uv --version

# create the .venv and install all dependencies (Python 3.10 + torch cu128)
uv sync

# verify imports, the CUDA 12.8 GPU, and the self-contained ops
uv run python scripts/check_environment.py
```

`uv sync` reads `.python-version` (3.10) and `pyproject.toml`. PyTorch / TorchVision
`+cu128` wheels come from the `pytorch-cu128` index; everything else from PyPI.

nuScenes data loading and the official tracking evaluation pull heavier, more
fragile transitive dependencies that are **not** needed for the unit tests or the
GPU smoke tests. Install them only when you actually use the dataset path:

```shell
uv sync --extra nuscenes
```

## Self-contained ops (replaces the old steps 4 & 5)

* **BEV IoU** (former CenterPoint `iou3d_nms_cuda`) → [`ops/iou3d_nms_cuda.py`](ops/iou3d_nms_cuda.py).
  Exposes `boxes_iou_bev_cpu` (exact shapely reference, used by data pre-processing /
  target assignment) and a vectorised `boxes_iou_bev` that runs on both CPU and CUDA
  tensors. No compilation step.
* **Detection NMS** (former SimpleTrack `mot_3d.preprocessing.nms`) →
  [`ops/simpletrack_nms.py`](ops/simpletrack_nms.py). Provides `nms` and
  `nu_array2mot_bbox`, reusing the `BBox` / `iou3d` primitives already vendored in
  `models/structures/boxes.py`.

## Tests

```shell
uv run pytest
```

The suite covers the BEV IoU op (CPU reference vs. vectorised torch, plus a GPU
path), the SimpleTrack NMS port, device resolution, the CUDA 12.8 build, and a
forward/backward pass of the GRAE model on the GPU. GPU tests are skipped
automatically when no CUDA device is present.

## Legacy conda setup (original PyTorch 1.9.0 / CUDA 11.1)

> Kept for reference. This configuration does **not** run on Blackwell GPUs and
> still relies on the external CenterPoint / SimpleTrack checkouts; prefer the uv
> setup above.

```shell
conda create -n grae3dmot python=3.8 -y
conda activate grae3dmot
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pandas==1.4.0 fvcore==0.1.5.post20221221 nuscenes-devkit matplotlib motmetrics==1.1.3
```

# Dataset preparation

**1. Download nuScenes**

Download the [nuScenes dataset](https://nuscenes.org/download)."


**2. Get nuScenes CenterPoint detection results**

Most existing MOT paper use [CenterPoint](https://github.com/tianweiy/CenterPoint) as public detection due to its better performance.

Following this [Github issue](https://github.com/tianweiy/CenterPoint/issues/249), you can download CenterPoint public detections that are provided by the authors:
- [With flip augmentation](https://mitprod-my.sharepoint.com/:f:/g/personal/tianweiy_mit_edu/Eip_tOTYSk5JhdVtVzlXlyABDPnGx9vsnwdo5SRK7bsh8w?e=vSdija) (68.5 NDS val performance)
- [Without flip augmentation](https://mitprod-my.sharepoint.com/:f:/g/personal/tianweiy_mit_edu/Er_nsH9Z2tRHnptBFJ_ompAByE3zu4E88xae691xyS6q_w?e=UqTmU2) (66.8 NDS val performance)

**3. Data pre-processing**

Please rename the json files with detection results for train, validation, and test as train.json, val.json and test.json.

Please save each file to its designated location.
```
center_point_det
├── train.json
├── val.json
├── test.json
```

Use this `script` to pre-process the detections (requires `uv sync --extra nuscenes`):
```
uv run python tools/convert_dataset.py
```
By default, `train.json` is used as the input. If you want to convert `validation.json`, please modify `mod="train"` to `mod="val"` in the main code of `convert_dataset.py`. (Note: This code is currently in beta, and we plan to allow users to customize arguments in the future.)


# Train & Eval

**1. Train**

The beta version of the code provides separate training implementations for single-GPU and multi-GPU setups.
```shell
# single GPU (defaults to config arch.args.device; override with --device)
uv run python single_gpu_train.py --device cuda:0

#multiple GPU
uv run python -m torch.distributed.launch --nproc_per_node=$GPU_NUM train.py
```

**2. Evaluation**

Evaluation can be performed using the trained parameters. Please refer to the official NuScenes website for the evaluation of the test set.
```shell
uv run python validation.py -r $CHECK_POINT_PATH -d cuda:0 -o $OUTPUT_PATH
```

# Results

We achieved the state-of-the art on nuScenes dataset.
<p align="center"> <img src='docs/test.png', height="280px", width="900px"> </p>  


----------------------------------
## 📢 News (2025-12-22)

We have released the **processed datasets** (train / validation / test) and the **model checkpoints (.pth) corresponding to the results reported in this paper** to facilitate research convenience and reproducibility.

All resources are publicly available at the following Google Drive link:  
https://drive.google.com/drive/folders/1V-rSzUsOtOY-hoxQET8r9unDnUudks4I?usp=sharing

## Dataset and Model Checkpoints

The released files include:
- Refined datasets for **train / validation / test**
- Model checkpoints (`.pth`) from the experiments reported in this paper

These materials are provided to support fair comparison and further research. If you find any bugs or issues, feel free to open an issue.

----------------------------------


**License**
###### Part of the code of project GRAE-3DMOT are taken from CenterPoint and 3DMOTFormer #####
###### Pytorch Template Project (https://github.com/victoresque/pytorch-template) #####
##### We plan to provide a detailed update on the license information  #####
