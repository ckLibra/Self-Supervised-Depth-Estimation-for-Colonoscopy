from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import numpy as np
import os
import torchvision.transforms as transforms
import math


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


def tensor2array(tensor):
    array = tensor.detach().cpu()
    #array = -(array.numpy())
    array = array.numpy()
    array = 0.5 + array * 0.5
    array = array.squeeze()

    return array


def test(model):
   #load data
    data_path = '/data12T/kcheng/colonoDepthEstimation/datasets/colon2depth'
    dirs = ["eval_A"]
    trans = transforms.Compose([transforms.Resize((512, 512), Image.BICUBIC),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                ])
    mean_relative_l1_error = 0.0
    mean_l1_error = 0.0
    rmse = 0.0
    num = 1.0
    for d in dirs:
        path = os.path.join(data_path, d)
        imgs_path = os.listdir(path)
        for img in imgs_path:
            img_path = os.path.join(path, img)
            frame = Image.open(img_path).convert('RGB')

            frame = trans(frame)
            frame = frame.unsqueeze(dim=0)
            frame = frame.to(torch.device('cuda'))

            generated = model(label=frame, inst=None, image=None, feat=None, infer=None, is_OC=True)

            #resize to gt size 256*256
            predicted = tensor2array(generated[0])
            predicted = Image.fromarray(predicted)
            predicted = predicted.resize((256, 256),Image.BICUBIC)
            predicted = np.array(predicted)

            name = img.split("FrameBuffer")
            depth_name = name[0] + "Depth" + name[1]
            depth_path = os.path.join("/data12T/kcheng/colonoDepthEstimation/datasets/colon2depth/eval_B", depth_name)
            #ground_truth = Image.open(depth_path).resize((512, 512),Image.BICUBIC)
            ground_truth = Image.open(depth_path)
            gt = np.array(ground_truth).astype(np.float32) / 255.0

            #mean relative l1-error
            l1_error = abs(predicted - gt)
            rel_error = l1_error[gt !=0] / gt[gt !=0]
            #print(np.mean(rel_error))
            mean_relative_l1_error = mean_relative_l1_error + (np.mean(rel_error) - mean_relative_l1_error) / num

            #mean l1-error
            mean_l1_error = mean_l1_error + (np.mean(l1_error) - mean_l1_error) / num

            #mean RMSE
            rmse = rmse + (math.sqrt(np.mean(l1_error * l1_error)) - rmse) / num

            num = num + 1

    return  mean_relative_l1_error, mean_l1_error, rmse


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()    
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
