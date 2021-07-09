#!/usr/bin/env python

import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image as Image
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from util.consistent_depth import consistency_mask


def gradient(matrix):
	#matrix is batch*c*h*w
	D_dy = matrix[:, :, 1:, :] - matrix[:, :, :-1, :]
	D_dx = matrix[:, :, :, 1:] - matrix[:, :, :, :-1]

	return D_dx, D_dy


def opencv_rainbow(resolution=256):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


def save_img(arr, name):
    save_path = '/data12T/kcheng/temp/depth'
    save_path = os.path.join(save_path, name)

    plt.imsave(save_path, arr, cmap=opencv_rainbow())


try:
	from .correlation import correlation # the custom cost volume layer
except:
	sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python

# end

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

#torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################
arguments_strModel = 'default'
"""
arguments_strModel = 'default'
arguments_strFirst = './images/first.png'
arguments_strSecond = './images/second.png'
arguments_strOut = './out.flo'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use
	if strOption == '--first' and strArgument != '': arguments_strFirst = strArgument # path to the first frame
	if strOption == '--second' and strArgument != '': arguments_strSecond = strArgument # path to the second frame
	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end
"""
##########################################################

backwarp_tenGrid = {}
backwarp_tenPartial = {}

def backwarp(tenInput, tenFlow):
	if str(tenFlow.shape) not in backwarp_tenGrid:
		tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
		tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

		backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
	# end

	if str(tenFlow.shape) not in backwarp_tenPartial:
		backwarp_tenPartial[str(tenFlow.shape)] = tenFlow.new_ones([ tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3] ])
	# end

	tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
	tenInput = torch.cat([ tenInput, backwarp_tenPartial[str(tenFlow.shape)] ], 1)

	tenOutput = torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)

	tenMask = tenOutput[:, -1:, :, :]; tenMask[tenMask > 0.999] = 1.0; tenMask[tenMask < 1.0] = 0.0

	return tenOutput[:, :-1, :, :] * tenMask
# end

##########################################################

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		class Extractor(torch.nn.Module):
			def __init__(self):
				super(Extractor, self).__init__()

				self.netOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
			# end

			def forward(self, tenInput):
				tenOne = self.netOne(tenInput)
				tenTwo = self.netTwo(tenOne)
				tenThr = self.netThr(tenTwo)
				tenFou = self.netFou(tenThr)
				tenFiv = self.netFiv(tenFou)
				tenSix = self.netSix(tenFiv)

				return [ tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix ]
			# end
		# end

		class Decoder(torch.nn.Module):
			def __init__(self, intLevel):
				super(Decoder, self).__init__()

				intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
				intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

				if intLevel < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.fltBackwarp = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

				self.netOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
				)
			# end

			def forward(self, tenFirst, tenSecond, objPrevious):
				tenFlow = None
				tenFeat = None

				if objPrevious is None:
					tenFlow = None
					tenFeat = None

					tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenFirst=tenFirst, tenSecond=tenSecond), negative_slope=0.1, inplace=False)

					tenFeat = torch.cat([ tenVolume ], 1)

				elif objPrevious is not None:
					tenFlow = self.netUpflow(objPrevious['tenFlow'])
					tenFeat = self.netUpfeat(objPrevious['tenFeat'])

					tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenFirst=tenFirst, tenSecond=backwarp(tenInput=tenSecond, tenFlow=tenFlow * self.fltBackwarp)), negative_slope=0.1, inplace=False)

					tenFeat = torch.cat([ tenVolume, tenFirst, tenFlow, tenFeat ], 1)

				# end

				tenFeat = torch.cat([ self.netOne(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netTwo(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netThr(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netFou(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netFiv(tenFeat), tenFeat ], 1)

				tenFlow = self.netSix(tenFeat)

				return {
					'tenFlow': tenFlow,
					'tenFeat': tenFeat
				}
			# end
		# end

		class Refiner(torch.nn.Module):
			def __init__(self):
				super(Refiner, self).__init__()

				self.netMain = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
				)
			# end

			def forward(self, tenInput):
				return self.netMain(tenInput)
			# end
		# end

		self.netExtractor = Extractor()

		self.netTwo = Decoder(2)
		self.netThr = Decoder(3)
		self.netFou = Decoder(4)
		self.netFiv = Decoder(5)
		self.netSix = Decoder(6)

		self.netRefiner = Refiner()

		self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-pwc/network-' + arguments_strModel + '.pytorch').items() })
	# end

	def forward(self, tenFirst, tenSecond):
		tenFirst = self.netExtractor(tenFirst)
		tenSecond = self.netExtractor(tenSecond)

		objEstimate = self.netSix(tenFirst[-1], tenSecond[-1], None)
		objEstimate = self.netFiv(tenFirst[-2], tenSecond[-2], objEstimate)
		objEstimate = self.netFou(tenFirst[-3], tenSecond[-3], objEstimate)
		objEstimate = self.netThr(tenFirst[-4], tenSecond[-4], objEstimate)
		objEstimate = self.netTwo(tenFirst[-5], tenSecond[-5], objEstimate)

		return objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])
	# end
# end

netNetwork = None

##########################################################

def estimate(tenFirst, tenSecond):
	global netNetwork

	if netNetwork is None:
		netNetwork = Network().cuda().eval()
	# end

	assert(tenFirst.shape[1] == tenSecond.shape[1])
	assert(tenFirst.shape[2] == tenSecond.shape[2])

	intWidth = tenFirst.shape[2]
	intHeight = tenFirst.shape[1]

	#assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	#assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

	tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
	tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight, intWidth)

	intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
	intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

	tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
	tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

	tenFlow = 20.0 * torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedFirst, tenPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
	tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

	return tenFlow[0, :, :, :].cpu()

num = 0
def warp_loss(tenFirst, tenSecond, depthFirst, depthSecond):
	with torch.no_grad():
		flowOutput = estimate(tenFirst, tenSecond)
		flowOutput_ = estimate(tenSecond, tenFirst)

	mask = consistency_mask(flowOutput.numpy().transpose(1, 2, 0),
							-flowOutput_.numpy().transpose(1, 2, 0),
							flowOutput.numpy().transpose(1, 2, 0), 1)

	mask_np = mask
	mask = torch.from_numpy(mask).cuda()

	#warp depthSecond to depthFirst
	of = flowOutput.numpy().transpose(1, 2, 0)
	x = numpy.linspace(0, of.shape[1] - 1, of.shape[1])
	y = numpy.linspace(0, of.shape[0] - 1, of.shape[0])
	X, Y = numpy.meshgrid(x, y)
	X = ((X + of[:, :, 0]) / of.shape[1] - 0.5) * 2
	Y = ((Y + of[:, :, 1]) / of.shape[0] - 0.5) * 2
	X = numpy.expand_dims(X, axis=2)
	Y = numpy.expand_dims(Y, axis=2)
	warp_grid = numpy.concatenate((X, Y), axis=2).astype(numpy.float32)

	flow_tensor = torch.tensor(warp_grid)
	flow_tensor = flow_tensor.unsqueeze(0).cuda()

	"""
	# gird sample,flow range from [-1,1],input size[n*c*h*w], gird size(n*h*w*2)
	warp_depth_first = torch.nn.functional.grid_sample(depthSecond, flow_tensor, mode='bilinear', padding_mode='border',
												       align_corners=False)

	if mask is not None:
		#normalize mask
		mask_weights_sum = torch.sum(mask.view(-1))
		mask_weights_sum = mask_weights_sum.float()
		mask_weights_sum = torch.clamp(mask_weights_sum, min=1)

	diff = warp_depth_first - depthFirst
	loss = (torch.sum((mask * diff.abs()).view(-1))) / mask_weights_sum
	"""

	#cal gradient
	source_depth_gx, source_depth_gy, target_depth_gx, target_depth_gy = [torch.FloatTensor(numpy.zeros((1, 1, 512, 512)).astype(numpy.float32)).cuda() for i in range(4)]
	source_depth_gx[:, :, :, :511], source_depth_gy[:, :, 0:511, :] = gradient(depthFirst)
	target_depth_gx[:, :, :, :511], target_depth_gy[:, :, 0:511, :] = gradient(depthSecond)

	warp_depth_gx = torch.nn.functional.grid_sample(target_depth_gx, flow_tensor, mode='bilinear', padding_mode='border', align_corners=False)
	warp_depth_gy = torch.nn.functional.grid_sample(target_depth_gy, flow_tensor, mode='bilinear', padding_mode='border', align_corners=False)

	diff_gx = source_depth_gx - warp_depth_gx
	diff_gy = source_depth_gy - warp_depth_gy
	if mask is not None:
		#normalize mask
		mask_weights_sum = torch.sum(mask.view(-1))
		mask_weights_sum = mask_weights_sum.float()
		mask_weights_sum = torch.clamp(mask_weights_sum, min=1)

	#edge-aware

	grad_img_x, grad_img_y = [torch.FloatTensor(numpy.zeros((1, 512, 512)).astype(numpy.float32)).cuda() for i in range(2)]

	grad_img_x[:, :, :511] = torch.mean(torch.abs(tenFirst[:, :, :-1] - tenFirst[:, :, 1:]), 0, keepdim=True)
	grad_img_y[:, :511, :] = torch.mean(torch.abs(tenSecond[:, :-1, :] - tenSecond[:, 1:, :]), 0, keepdim=True)

	diff_gx *= torch.exp(-grad_img_x)
	diff_gy *= torch.exp(-grad_img_y)

	loss = (torch.sum((mask * diff_gx.abs()).view(-1)) + torch.sum((mask * diff_gy.abs()).view(-1))) / mask_weights_sum


	"""
	global num
	if num < 100:
		#first_gx = ((source_depth_gx).squeeze().detach().cpu().numpy() * 0.5 + 0.5)
		#warp_gx = ((warp_depth_gx).squeeze().detach().cpu().numpy() * 0.5 + 0.5)
		first_depth_pic = (depthFirst.squeeze().detach().cpu().numpy() * 0.5 + 0.5) * mask_np

		save_img(first_depth_pic, str(num).zfill(4) + 'd_first.png')
		#save_img(first_gx, str(num).zfill(4) + 'g_warp.png')
		#save_img(warp_gx, str(num).zfill(4) + 'g_first.png')
	num = num + 1

	global num
	first_depth_pic = (depthFirst.squeeze().detach().cpu().numpy() * 0.5 + 0.5)
	warp_first_depth_pic = (warp_depth_first.squeeze().detach().cpu().numpy() * 0.5 + 0.5)
	save_img(warp_first_depth_pic, str(num) + '_warp.png')
	save_img(first_depth_pic, str(num) + '_first.png')
	num = num + 1

	global num
	tenSecond = tenSecond.unsqueeze(0)
	tenSecond = tenSecond.cuda()
	warp_frame_first = torch.nn.functional.grid_sample(tenSecond, flow_tensor, mode='bilinear', padding_mode='zeros',
												       align_corners=False)
	warp_first_depth_pic = (warp_depth_first.squeeze().detach().cpu().numpy() * 0.5 + 0.5)
	warp_first_frame_pic = warp_frame_first.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255
	second_depth_pic = (depthSecond.squeeze().detach().cpu().numpy() * 0.5 + 0.5)
	first_depth_pic = (depthFirst.squeeze().detach().cpu().numpy() * 0.5 + 0.5)
	frame1 = tenFirst.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255
	frame2 = tenSecond.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255


	#im1 = Image.fromarray(numpy.uint8(warp_first_depth_pic))
	#im1.save("/data12T/kcheng/temp/s/" + "warpDepth" + str(num) + ".png")
	save_img(warp_first_depth_pic, str(num)+'_warp.png')

	#im4 = Image.fromarray(numpy.uint8(second_depth_pic))
	#im4.save("/data12T/kcheng/temp/s/"+"Depth_" + str(num) + "_2.png")
	save_img(second_depth_pic, str(num) + '_second.png')

	#im2 = Image.fromarray(numpy.uint8(first_depth_pic))
	#im2.save("/data12T/kcheng/temp/s/"+"Depth_" + str(num) + "_1.png")
	save_img(first_depth_pic, str(num) + '_first.png')

	im3 = Image.fromarray(numpy.uint8(frame1)[:, :, ::-1])
	#im3.save("/data12T/kcheng/temp/"+"frame/" + str(num) + "_1.png")
	im5 = Image.fromarray(numpy.uint8(frame2)[:, :, ::-1])
	#im5.save("/data12T/kcheng/temp/"+"frame/" + str(num) + "_2.png")
	im6 = Image.fromarray(numpy.uint8(warp_first_frame_pic)[:, :, ::-1])
	#im6.save("/data12T/kcheng/temp/"+"frame/" + str(num) + "_warp.png")
	num = num + 1
	"""

	return loss


def warp_depth(tenFirst, tenSecond, depthFirst, depthSecond):
	with torch.no_grad():
		flowOutput = estimate(tenFirst, tenSecond)
		flowOutput_ = estimate(tenSecond, tenFirst)

	mask = consistency_mask(flowOutput.numpy().transpose(1, 2, 0),
							-flowOutput_.numpy().transpose(1, 2, 0),
							flowOutput.numpy().transpose(1, 2, 0), 1)

	mask = torch.from_numpy(mask).cuda()

	#warp depthSecond to depthFirst
	of = flowOutput.numpy().transpose(1, 2, 0)
	x = numpy.linspace(0, of.shape[1] - 1, of.shape[1])
	y = numpy.linspace(0, of.shape[0] - 1, of.shape[0])
	X, Y = numpy.meshgrid(x, y)
	X = ((X + of[:, :, 0]) / of.shape[1] - 0.5) * 2
	Y = ((Y + of[:, :, 1]) / of.shape[0] - 0.5) * 2
	X = numpy.expand_dims(X, axis=2)
	Y = numpy.expand_dims(Y, axis=2)
	warp_grid = numpy.concatenate((X, Y), axis=2).astype(numpy.float32)

	flow_tensor = torch.tensor(warp_grid)
	flow_tensor = flow_tensor.unsqueeze(0).cuda()


	# gird sample,flow range from [-1,1],input size[n*c*h*w], gird size(n*h*w*2)
	warp_depth_first = torch.nn.functional.grid_sample(depthSecond, flow_tensor, mode='bilinear', padding_mode='border',
												       align_corners=False)

	depthFirst = depthFirst * mask
	valid_mask = depthFirst > 0.005

	diff = ((warp_depth_first - depthFirst).abs())[valid_mask]

	loss = torch.mean(diff)

	return loss


def warp_gradient(tenFirst, tenSecond, depthFirst, depthSecond):
	with torch.no_grad():
		flowOutput = estimate(tenFirst, tenSecond)
		flowOutput_ = estimate(tenSecond, tenFirst)

	mask = consistency_mask(flowOutput.numpy().transpose(1, 2, 0),
							-flowOutput_.numpy().transpose(1, 2, 0),
							flowOutput.numpy().transpose(1, 2, 0), 1)

	mask_np = mask
	mask = torch.from_numpy(mask).cuda()

	#warp depthSecond to depthFirst
	of = flowOutput.numpy().transpose(1, 2, 0)
	x = numpy.linspace(0, of.shape[1] - 1, of.shape[1])
	y = numpy.linspace(0, of.shape[0] - 1, of.shape[0])
	X, Y = numpy.meshgrid(x, y)
	X = ((X + of[:, :, 0]) / of.shape[1] - 0.5) * 2
	Y = ((Y + of[:, :, 1]) / of.shape[0] - 0.5) * 2
	X = numpy.expand_dims(X, axis=2)
	Y = numpy.expand_dims(Y, axis=2)
	warp_grid = numpy.concatenate((X, Y), axis=2).astype(numpy.float32)

	flow_tensor = torch.tensor(warp_grid)
	flow_tensor = flow_tensor.unsqueeze(0).cuda()

	#cal gradient
	source_depth_gx, source_depth_gy, target_depth_gx, target_depth_gy = [torch.FloatTensor(numpy.zeros((1, 1, 512, 512)).astype(numpy.float32)).cuda() for i in range(4)]
	source_depth_gx[:, :, :, :511], source_depth_gy[:, :, 0:511, :] = gradient(depthFirst)
	target_depth_gx[:, :, :, :511], target_depth_gy[:, :, 0:511, :] = gradient(depthSecond)

	warp_depth_gx = torch.nn.functional.grid_sample(target_depth_gx, flow_tensor, mode='bilinear', padding_mode='border', align_corners=False)
	warp_depth_gy = torch.nn.functional.grid_sample(target_depth_gy, flow_tensor, mode='bilinear', padding_mode='border', align_corners=False)

	source_depth_gx = source_depth_gx * mask
	mask_valid = source_depth_gx > 0.001

	diff_gx = (source_depth_gx - warp_depth_gx).abs()[mask_valid]
	#diff_gy = (source_depth_gy - warp_depth_gy).abs() / source_depth_gy


	loss = torch.mean((diff_gx).view(-1))

	return loss