import torch
from oflow.optical_flow import estimate
from util.consistent_depth import consistency_mask
from PIL import Image
import numpy as np

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def gradient(matrix):
    # matrix is batch*c*h*w
    D_dy = matrix[1:, :] - matrix[:-1, :]
    D_dx = matrix[:, 1:] - matrix[:, :-1]

    return D_dx, D_dy


img1 = np.array(Image.open('img1.jpg'))
img2 = np.array(Image.open('img2.jpg'))
d1 = np.array(Image.open('d1.jpg').convert('L')).astype(np.int32)
d2 = np.array(Image.open('d2.jpg').convert('L')).astype(np.int32)

d1_dx, d1_dy = gradient(d1)
d2_dx, d2_dy = gradient(d2)

Image.fromarray(np.abs(d1_dx)*127).convert('L').save('d1_dx.jpg')
Image.fromarray(np.abs(d1_dy)*127).convert('L').save('d1_dy.jpg')
Image.fromarray(np.abs(d2_dx)*127).convert('L').save('d2_dx.jpg')
Image.fromarray(np.abs(d2_dy)*127).convert('L').save('d2_dy.jpg')

tenFirst = torch.FloatTensor(
    np.ascontiguousarray(img1[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
tenSecond = torch.FloatTensor(
    np.ascontiguousarray(img2[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))

with torch.no_grad():
    flowOutput = estimate(tenFirst, tenSecond)
    flowOutput_ = estimate(tenSecond, tenFirst)

mask = consistency_mask(flowOutput.numpy().transpose(1, 2, 0),
                        -flowOutput_.numpy().transpose(1, 2, 0),
                        flowOutput.numpy().transpose(1, 2, 0), 1)

mask_img = Image.fromarray(mask)

d2_dx = d2_dx * mask[:, :255]
d2_dy = d2_dy * mask[:255, :]
d2_dx = Image.fromarray(np.abs(d2_dx)*127).convert('L').save('d2_dx_.jpg')
d2_dy = Image.fromarray(np.abs(d2_dy)*127).convert('L').save('d2_dy_.jpg')
mask_img.save('mask.jpg')
