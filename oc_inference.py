import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
import math


def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0


from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

from data.oc_dataset import OC_Dataset
from oflow.optical_flow import warp_loss
import torchvision.transforms as transforms


opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10


# OC dataset
depth_transform = transforms.Compose([#transforms.Resize((512, 512), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
oc_dataset = OC_Dataset(root='/data2/kcheng/oc_dataset_test', depth_transform=depth_transform)
oc_dataloader = torch.utils.data.DataLoader(dataset=oc_dataset, batch_size=1, shuffle=False, num_workers=1)
oc_dataiter = iter(oc_dataloader)
mean_warp_loss = 0.0
oc_num = 1.0
mwl = 0.0024
var = 0.0
max = 0
min = 1
print(mwl)

model = create_model(opt)
visualizer = Visualizer(opt)
if opt.fp16:
    from apex import amp

    model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D], opt_level='O1')
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
else:
    optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D


while True:
    #############OC training##############
    try:
        source_for_depth, target_for_depth, source_for_op, target_for_op = oc_dataiter.next()
    except StopIteration:
        #to normalize to the true depth unit, every metric * 10
        print("warp_loss: " + str((mean_warp_loss)))
        print(math.sqrt(var))
        mean_warp_loss = 0.0
        oc_num = 1.0
        print(max)
        print(min)
        exit()

    oc_warp_loss_weight = 1.0
    with torch.no_grad():
        generated_oc_depth = model(label=torch.stack([source_for_depth[0], target_for_depth[0]], dim=0), inst=None,
                                   image=None, feat=None, infer=True, is_OC=True)

    oc_warp_loss = oc_warp_loss_weight * warp_loss(source_for_op[0], target_for_op[0],
                                                   generated_oc_depth[0].unsqueeze(0),
                                                   generated_oc_depth[1].unsqueeze(0)).detach().item()
    #print(oc_warp_loss)
    max = oc_warp_loss if oc_warp_loss > max else max
    min = oc_warp_loss if oc_warp_loss < min else min
    var = var + ((oc_warp_loss - mwl) * (oc_warp_loss - mwl) - var) / oc_num
    mean_warp_loss = mean_warp_loss + (oc_warp_loss - mean_warp_loss) / oc_num
    #print(mean_warp_loss)
    oc_num = oc_num + 1
    print(oc_num)

