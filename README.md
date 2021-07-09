
<br><br><br><br>

# Depth Estimation for Colonoscopy Images with Self-supervised Learning from Videos
## Abstract
Depth estimation in colonoscopy images provides geometric
clues for downstream medical analysis tasks, such as polyp detection,
3D reconstruction, and diagnosis. Recently, deep learning technology
has made signicant progress in monocular depth estimation for nat-
ural scenes. However, without sufficient ground truth of dense depth
maps for colonoscopy images, it is signicantly challenging to train deep
neural networks for colonoscopy depth estimation. In this paper, we pro-
pose a novel approach that makes full use of both synthetic data and real
colonoscopy videos.We use synthetic data with ground truth depth maps
to train a depth estimation network with a generative adversarial network
model. Despite the lack of ground truth depth, real colonoscopy videos
are used to train the network in a self-supervision manner by exploiting 
temporal consistency between neighboring frames. Furthermore, we
design a masked gradient warping loss in order to ensure temporal consis-
tency with more reliable correspondences. We conducted both quantita-
tive and qualitative analysis on an existing synthetic dataset and a set of
real colonoscopy videos, demonstrating the superiority of our method on
more accurate and consistent depth estimation for colonoscopy images.

[In MICCAI 2021].  

## Prerequisites
- Linux
- Python 3, pytorch1.4
- NVIDIA GPU (11G memory or larger) + CUDA cuDNN

### Testing
- Test the model (`bash ./scripts/test.sh`):
```bash
#!./scripts/test.sh
python syn_test.py --name colon2depth_512p --no_instance --label_nc 0
```

### Dataset
- For synthetic data, you can download from http://cmic.cs.ucl.ac.uk/ColonoscopyDepth/.
- For real dataset, you can download from https://github.com/dashishi/LDPolypVideo-Benchmark. This is our raw data, the data used for training is the subset of it. We crop the raw image to remove the black corner.

### Training
- Train a model at 512 x 512 resolution (`bash ./scripts/train.sh`):
```bash
#!./scripts/train_512p.sh
python train.py --name label2city_512p
```
- To view training results, please checkout intermediate results in `./checkpoints/label2city_512p/web/index.html`.
If you have tensorflow installed, you can see tensorboard logs in `./checkpoints/label2city_512p/logs` by adding `--tf_log` to the training scripts.

### Multi-GPU training
- Train a model using multiple GPUs (`bash ./scripts/train_512p_multigpu.sh`):
```bash
#!./scripts/train_512p_multigpu.sh
python train.py --name colon2depth_512p --batchSize 8 --gpu_ids 1,2 --label_nc 0 --no_instance --tf_log --no_vgg_loss --continue_train```
Note: this is not tested and we trained our model using single GPU only. Please use at your own discretion.

## Citation

If you find this useful for your research, please use the following.

```
TBD
```

## Acknowledgments
This code borrows from [NVIDIA/pix2pixHD](https://github.com/NVIDIA/pix2pixHD).
