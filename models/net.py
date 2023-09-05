import torch
import torch.nn as nn

def conv_bn(inp, oup, s=1, nlin_layer=nn.LeakyReLU):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 3, s, 1, bias=False)
		nn.BatchNorm2d(oup)
		nlin_layer(inplace=True)
	)

def conv_1x1_bn(inp, oup, s=1, leaky=0):
	return nn.Sequntial(
		nn.Conv2d(inp, oup, 1, s, 0, bias=False)
		nn.BatchNorm2d(oup),
		nn.LeakyReLU(negative_slope=leaky, inplace=True)
	)

class HSigmoid(nn.Module):
	def __init__(self, inplace=False):
		super(HSigmoid, self).__init__()
		self.relu = nn.ReLU6(inplace=inplace)
	def forward(self, x):
		return self.relu(x+3.)/6

class HSwish(nn.Module):
	def __init__(self, inplace=False):
		super(HSwish, self).__init__()
		self.hsigmoid = HSigmoid(inplace)
	def forward(self, x):
		return x*self.hsigmoid(x)


class MobileNetV3(nn.Module):
	def __init__(self):
		super(MobileNetV3, self).__init__()
		self.