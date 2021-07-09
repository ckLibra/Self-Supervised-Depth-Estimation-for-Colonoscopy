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

def tensor2array(tensor):
    array = tensor.detach().cpu()
    #array = -(array.numpy())
    array = array.numpy()
    array = 0.5 + array * 0.5
    array = array.squeeze()

    return array


opt = TestOptions().parse(save=False)
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

print("loading model...")
# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)

    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx

print("loading data")
#load data
data_path = '/data12T/kcheng/colonoscopy/colonoDepthEstimation/datasets/UCLDepth3'
#dirs = ["test_A", "train_A"]
dirs = ["test_A"]
trans = transforms.Compose([transforms.Resize((512, 512), Image.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ])
print("inferencing")
mean_relative_l1_error = 0.0
mean_l1_error = 0.0
rmse = 0.0
max_rle = 0.0
min_rle = 256.0
max_le = 0.0
min_le = 256.0
max_rmse = 0.0
min_rmse = 256.0
num = 1.0
max_path = None
#####################################
mean_mrle = 0.015514
mean_mle = 0.032781
var_mrle = 0.0
var_mle = 0.0
mrle_set = dict()
for d in dirs:
    path = os.path.join(data_path, d)
    imgs_path = os.listdir(path)
    for img in imgs_path:
        img_path = os.path.join(path, img)
        frame = Image.open(img_path).convert('RGB')
        #frame = Image.open("/data12T/kcheng/colonoDepthEstimation/datasets/colon2depth/test_A/T3_L1_1_resized_FrameBuffer_0294.png").convert('RGB')

        frame = trans(frame)
        frame = frame.unsqueeze(dim=0)
        frame = frame.to(torch.device('cuda'))

        generated = model.inference(frame, None, None)

        #resize to gt size 256*256
        predicted = tensor2array(generated[0])
        print(predicted.shape)
        predicted = Image.fromarray(predicted)
        predicted = predicted.resize((256, 256), Image.BICUBIC)
        predicted = np.array(predicted)
        print(predicted)
        exit()

        name = img.split("FrameBuffer")
        depth_name = name[0] + "Depth" + name[1]
        depth_path = os.path.join("/data12T/kcheng/colonoscopy/colonoDepthEstimation/datasets/UCLDepth3/test_B", depth_name)
        #ground_truth = Image.open(depth_path).resize((512, 512),Image.BICUBIC)
        #ground_truth = Image.open(depth_path)
        #ground_truth = Image.open("/data12T/kcheng/colonoDepthEstimation/datasets/colon2depth/test_B/T3_L1_1_resized_Depth_0294.png")
        #gt = np.array(ground_truth).astype(np.float32) / 255.0
        gt = plt.imread(depth_path).astype(np.float32)

        #mean relative l1-error
        l1_error = abs(predicted - gt)[gt > 0]

        rel_error = l1_error / gt[gt > 0]
        #print(np.mean(rel_error))
        mean_relative_l1_error = mean_relative_l1_error + (np.mean(rel_error) - mean_relative_l1_error) / num
        var_mrle = var_mrle + ((np.mean(rel_error) - mean_mrle)*(np.mean(rel_error) - mean_mrle) - var_mrle) / num
        mrle_set[depth_name] = np.mean(rel_error)

        #mean l1-error
        mean_l1_error = mean_l1_error + (np.mean(l1_error) - mean_l1_error) / num
        var_mle = var_mle + ((np.mean(l1_error)*20 - mean_mle) * (np.mean(l1_error)*20 - mean_mle) - var_mle) / num

        #mean RMSE
        rmse_temp = math.sqrt(np.mean(l1_error * l1_error))
        rmse = rmse + (rmse_temp - rmse) / num

        num = num + 1

        #max,min
        min_le = min_le if np.mean(l1_error) > min_le else np.mean(l1_error)
        max_le = max_le if np.mean(l1_error) < max_le else np.mean(l1_error)
        min_rle = min_rle if np.mean(rel_error) > min_rle else np.mean(rel_error)
        max_path = img_path if np.mean(rel_error) > max_rle else max_path
        max_rle = max_rle if np.mean(rel_error) < max_rle else np.mean(rel_error)
        min_rmse = min_rmse if rmse_temp > min_rmse else rmse_temp
        max_rmse = max_rmse if rmse_temp < max_rmse else rmse_temp
        #save_img(generated[0], d, img)
        if num % 100 == 0:
            print("----------------------------------------------------------------------")
            print('process image... %s mean l1 error...%f' %(img_path, mean_l1_error*20))
            print('process image... %s mean_rel_l1_error...%f' %(img_path, mean_relative_l1_error))
            print('process image... %s rmse...%f' %(img_path, rmse*20))
            print('min_le: %f max_le:%f min_rle:%f max_rle:%f min_rmse:%f max_rmse:%f' %(min_le*20,
                                                    max_le*20, min_rle, max_rle, min_rmse*20, max_rmse*20))
            print(max_path)
            print("----------------------------------------------------------------------")

print('mle root variance: %f' % math.sqrt(var_mle))
print('mrle root variance: %f' % math.sqrt(var_mrle))

#sorted_mrle_arr = sorted(mrle_set.items(), key=lambda x: x[1], reverse=True)
#print(sorted_mrle_arr[:50])

