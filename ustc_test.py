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
import cv2


def visualize_depth_map(title, depth_maps, min_value_=None, max_value_=None, idx=None, color_mode=cv2.COLORMAP_JET):
    min_value_list = []
    max_value_list = []

    if idx is None:
        for i in range(depth_maps.shape[0]):
            depth_map_cpu = depth_maps[i].data.cpu().numpy()
            if min_value_ is None and max_value_ is None:

                min_value = np.min(depth_map_cpu)
                max_value = np.max(depth_map_cpu)
                min_value_list.append(min_value)
                max_value_list.append(max_value)
            else:
                min_value = min_value_[i]
                max_value = max_value_[i]

            depth_map_cpu = np.moveaxis(depth_map_cpu, source=[0, 1, 2], destination=[2, 0, 1])
            depth_map_visualize = np.abs((depth_map_cpu - min_value) / (max_value - min_value) * 255)
            depth_map_visualize[depth_map_visualize > 255] = 255
            depth_map_visualize[depth_map_visualize <= 0.0] = 0
            depth_map_visualize = cv2.applyColorMap(np.uint8(depth_map_visualize), color_mode)
            # cv2.imshow(title + "_" + str(i), depth_map_visualize)
            cv2.imwrite(title, depth_map_visualize)

        return min_value_list, max_value_list
    else:
        for id in idx:
            depth_map_cpu = depth_maps[id].data.cpu().numpy()
            if min_value_ is None and max_value_ is None:
                min_value = np.min(depth_map_cpu)
                max_value = np.max(depth_map_cpu)
                min_value_list.append(min_value)
                max_value_list.append(max_value)
            else:
                min_value = min_value_[id]
                max_value = max_value_[id]

            depth_map_cpu = np.moveaxis(depth_map_cpu, source=[0, 1, 2], destination=[2, 0, 1])
            depth_map_visualize = np.abs((depth_map_cpu - min_value) / (max_value - min_value) * 255)
            depth_map_visualize[depth_map_visualize > 255] = 255
            depth_map_visualize[depth_map_visualize <= 0.0] = 0
            depth_map_visualize = cv2.applyColorMap(np.uint8(depth_map_visualize), color_mode)
            # cv2.imshow(title + "_" + str(id), depth_map_visualize)
            cv2.imwrite(title, depth_map_visualize)

        return min_value_list, max_value_list


def opencv_rainbow(resolution=256):
    # Construct the opencv equivalent of Rainbow,
    """
    ucl_color_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )
    """
    """
    colormap_jet 
    """
    """
    ucl_color_data = (
        (0.000, (0.60, 0.00, 1.00)),
        (0.400, (0.00, 0.00, 1.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (1.00, 1.00, 0.00)),
        (1.000, (1.00, 0.00, 0.00))
    )
    """

    ucl_color_data = (
        (0.000, (1.00, 1.00, 0.00)),
        (0.400, (0.20, 0.67, 0.51)),
        (0.600, (0.20, 0.42, 0.54)),
        (0.800, (0.20, 0.19, 0.47)),
        (1.000, (0.25, 0.02, 0.34))
    )

    """
    ucl_color_data = (
        (0.000, (0.16, 0.04, 1.00)),
        (0.100, (0.16, 0.50, 0.80)),
        (0.178, (0.47, 1.00, 0.04)),
        (1.000, (1.00, 0.00, 0.00))
    )
    """

    return LinearSegmentedColormap.from_list('opencv_rainbow', ucl_color_data, resolution)


def save_img(arr, name):
    save_path = '/data12T/kcheng/colonoscopy/predict/ustc_sample'
    #save_path_img = '/data12T/kcheng/colonoscopy/predict/CVC-video_visualization'
    temp_dir = name.split('/')[-2]
    save_path = os.path.join(save_path, name.split('/')[-2])
    #save_path_img = os.path.join(save_path_img, name.split('/')[-2])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        #os.makedirs(save_path_img)
    save_path = os.path.join(save_path, name.split('/')[-1])


    #save_path_img = os.path.join(save_path_img, name.split('/')[-1])
    #save_path_img = save_path_img.split('.')[0] + '.png'
    #save_path = save_path.split('.')[0]

    arr = arr.detach().cpu()
    arr = arr.numpy()
    arr = 0.5 + arr * 0.5

    #full_arr = np.zeros((288, 384), dtype=np.float)
    #full_arr_img = np.zeros((288, 384), dtype=np.float)
    arr = np.squeeze(arr)
    #arr_img = np.squeeze(arr_img)

    """
    if temp_dir in d1:
        arr = np.array(Image.fromarray(arr).resize((271, 252), Image.BICUBIC))
        arr_img = np.array(Image.fromarray(arr_img).resize((271, 252), Image.BICUBIC))

        full_arr[28:280, 72:343] = arr
        full_arr_img[28:280, 72:343] = arr_img
    else:
        arr = np.array(Image.fromarray(arr).resize((287, 265), Image.BICUBIC))
        arr_img = np.array(Image.fromarray(arr_img).resize((287, 265), Image.BICUBIC))

        full_arr[15:280, 56:343] = arr
        full_arr_img[15:280, 56:343] = arr_img

    np.save(save_path, full_arr)
    #arr_img = np.array(Image.fromarray(arr_img))
    """
    # open gt
    """
    gt_path = os.path.join('/data2/kcheng/gt', name.split('/')[-1])
    gt = plt.imread(gt_path).astype(np.float32)
    mask = gt > 0.01

    error = abs(arr - gt) * mask * 20.0
    #error = 10 ** error

    print(np.mean(error))
    #error = (error) * (error)
    """
    plt.imsave(save_path, arr, cmap=opencv_rainbow(), vmin=0, vmax=0.5)
    np.save(save_path[:-3]+'npy', arr)
    #plt.imsave(save_path_img, full_arr_img, cmap=opencv_rainbow())
    # plt.imsave(save_path, error, cmap=plt.cm.get_cmap('winter'), vmin=0, vmax=3.8)
    # plt.imsave(save_path, arr, cmap=opencv_rainbow(), vmin=0, vmax=3.8)


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
# load data
data_path = '/data12T/kcheng/real_cases/'
# data_path = '/data12T/kcheng/colonoDepthEstimation/datasets/colon2depth/1'
dirs = os.listdir(data_path)
trans = transforms.Compose([  # transforms.Resize((512, 512), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
print("inferencing")
# path = data_path
for d in dirs:
    path = os.path.join(data_path, d)
    imgs_path = os.listdir(path)
    for img in imgs_path:
        img_path = os.path.join(path, img)

        frame = Image.open(img_path).convert('RGB').resize((512, 512), Image.BICUBIC)
        frame = np.array(frame)

        frame = trans(frame)
        frame = frame.unsqueeze(dim=0)
        frame = frame.to(torch.device('cuda'))

        generated = model.inference(frame, None, None)

        """
        if not os.path.exists('/data12T/kcheng/predict/ustc/'):
            os.makedirs('/data12T/kcheng/predict/ustc/')
        save_img_path = '/data12T/kcheng/predict/ucl/' + img.split('.')[0] + '.png'
    
        visualize_depth_map(save_img_path, generated)
        """

        save_img(generated[0], img_path)
        print('process image... %s' % img_path)
