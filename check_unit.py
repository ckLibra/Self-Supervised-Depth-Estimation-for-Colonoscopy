import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

im = plt.imread("test.png")
im = np.array(im)*20.0

im2 = Image.open("test.png")

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
im2 = trans(im2)
im2 = (im2.numpy() * 0.5 + 0.5) * 20.0

print(im2 - im < 0.000001)


