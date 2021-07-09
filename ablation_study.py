import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import torchvision.transforms as transforms

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from oflow.optical_flow import warp_loss, warp_depth, warp_gradient

trans_depth = transforms.Compose([transforms.Resize((512, 512), Image.BICUBIC),
                            transforms.ToTensor()
                            ])

trans_img = transforms.Compose([transforms.Resize((512, 512), Image.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ])

pathA = '/data12T/kcheng/colonoscopy/colonoDepthEstimation/datasets/colon2depth/train_A'
pathB = '/data12T/kcheng/colonoscopy/colonoDepthEstimation/datasets/colon2depth/train_B'
imgs = sorted(os.listdir(pathA))
depths = sorted(os.listdir(pathB))
depth_loss = 0.0
gradient_loss = 0.0
depth_variance = 0.0
gradient_variance = 0.0
for iter in range(len(imgs)-2):
    if int(imgs[iter][-8:-4])+1 != int(imgs[iter+1]):
    img_path1 = os.path.join(pathA, imgs[iter])
    img_path2 = os.path.join(pathA, imgs[iter+1])
    depth_path1 = os.path.join(pathB, depths[iter])
    depth_path2 = os.path.join(pathB, depths[iter+1])
    depth1 = Image.open(depth_path1)
    depth2 = Image.open(depth_path2)
    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2)

    # frame = Image.open("/data12T/kcheng/colonoDepthEstimation/datasets/colon2depth/test_A/T3_L1_1_resized_FrameBuffer_0294.png").convert('RGB')

    img1 = trans_img(img1)
    img1 = img1.to(torch.device('cuda'))

    img2 = trans_img(img2)
    img2 = img2.to(torch.device('cuda'))

    depth1 = trans_depth(depth1)
    depth1 = depth1.unsqueeze(dim=0)
    depth1 = depth1.to(torch.device('cuda'))

    depth2 = trans_depth(depth2)
    depth2 = depth2.unsqueeze(dim=0)
    depth2 = depth2.to(torch.device('cuda'))

    depth_loss = depth_loss + (warp_gradient(img1, img2, depth1, depth2).detach().cpu().numpy() - depth_loss) / (iter+1)

    #variance = variance + ((loss - 0) * (loss - 0) - variance) / (iter + 1)
    print(depth_loss)

