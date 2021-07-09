import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import time
import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

from data.oc_dataset import OC_Dataset
from oflow.optical_flow import warp_loss
from util.util import test
import torchvision.transforms as transforms

"""
#set random
seed = 2020
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
#torch.set_deterministic(True)
print('set random seed ' + str(seed))
"""

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

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

finetune_epoch = []
#finetune_epoch = [0, 1]
mrle = []
mle = []
rmse = []
oc_loss = []
oc_epoch = []

#OC dataset
oc_batch = 4
oc_warp_loss_weight = 5.0
depth_transform = transforms.Compose([#transforms.Resize((512, 512), Image.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ])
oc_dataset = OC_Dataset(root='/data12T/kcheng/colonoscopy/oc_dataset_train', depth_transform=depth_transform)
oc_dataloader = torch.utils.data.DataLoader(dataset=oc_dataset, batch_size=oc_batch, shuffle=True, num_workers=1, drop_last=True)
oc_dataiter = iter(oc_dataloader)
mean_warp_loss = 0.0
oc_num = 1.0
oc_count = 0

model = create_model(opt)
visualizer = Visualizer(opt)
if opt.fp16:    
    from apex import amp
    model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D], opt_level='O1')             
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
else:
    optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

#for finetune, need update lr first
print(opt.lr)
lrd = opt.lr / opt.niter_decay
lr = opt.lr - lrd * (start_epoch - opt.niter - 1)
for param_group in model.module.optimizer_D.param_groups:
    param_group['lr'] = lr
for param_group in model.module.optimizer_G.param_groups:
    param_group['lr'] = lr
model.module.old_lr = lr
print("-----------------------------------")
print(lr)
print("-----------------------------------")

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        #VC forward
        losses, generated = model(Variable(data['label']), Variable(data['inst']),
            Variable(data['image']), Variable(data['feat']), infer=save_fake, is_OC=False)

        # sum per device losses
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + 2*loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)
        #loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)

        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        if opt.fp16:                                
            with amp.scale_loss(loss_G, optimizer_G) as scaled_loss: scaled_loss.backward()                
        else:
            loss_G.backward()          
        optimizer_G.step()

        # update discriminator weights
        if i % 5 == 0:
            optimizer_D.zero_grad()
            if opt.fp16:
                with amp.scale_loss(loss_D, optimizer_D) as scaled_loss: scaled_loss.backward()
            else:
                loss_D.backward()
            optimizer_D.step()

        #############OC training##############
        if epoch > 70:
            try:
                source_for_depth, target_for_depth, source_for_op, target_for_op = oc_dataiter.next()
            except StopIteration:
                oc_dataiter = iter(oc_dataloader)
                source_for_depth, target_for_depth, source_for_op, target_for_op = oc_dataiter.next()

                oc_loss.append(mean_warp_loss / oc_warp_loss_weight)
                oc_epoch.append(oc_count)
                print("warp_loss: " + str(mean_warp_loss))
                mean_warp_loss = 0.0
                oc_num = 1.0
                oc_count = oc_count + 1

                oc_warp_loss_weight = oc_warp_loss_weight + 2.0
                print("warp weight: " + str(oc_warp_loss_weight))

            oc_warp_loss = 0.0
            for oc_index in range(oc_batch):
                generated_oc_depth = model(label=torch.stack([source_for_depth[oc_index], target_for_depth[oc_index]], dim=0), inst=None, image=None, feat=None, infer=save_fake, is_OC=True)
                oc_warp_loss += warp_loss(source_for_op[oc_index], target_for_op[oc_index], generated_oc_depth[0].unsqueeze(0), generated_oc_depth[1].unsqueeze(0))

            oc_warp_loss = oc_warp_loss_weight * oc_warp_loss / oc_batch

            optimizer_G.zero_grad()
            oc_warp_loss.backward()
            optimizer_G.step()

            mean_warp_loss = mean_warp_loss + (oc_warp_loss.detach().item() - mean_warp_loss) / oc_num
            oc_num = oc_num + 1


        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

        ### display output images
        if save_fake:
            visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                   ('synthesized_image', util.tensor2im(generated.data[0])),
                                   ('real_image', util.tensor2im(data['image'][0]))])
            visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break
       
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()

    ### test
    if epoch % 5 == 0:
        mean_relative_l1_error, mean_l1_error, r_mean_square_error = test(model)
        mrle.append(mean_relative_l1_error)
        mle.append(mean_l1_error)
        rmse.append(r_mean_square_error)
        finetune_epoch.append(epoch)

    plt.plot(finetune_epoch, mrle)
    #plt.plot(finetune_epoch, mle)
    #plt.plot(finetune_epoch, rmse)
    plt.title('eval result')
    plt.savefig('eval_result.jpg')
    plt.close()

plt.plot(oc_epoch, oc_loss)
plt.title('oc_loss')
plt.savefig('oc_loss.jpg')

