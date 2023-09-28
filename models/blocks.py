import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv_bn(inp, oup, s=1, nl='RE', leaky=0):
	if nl=='RE':
		nlin_layer = nn.LeakyReLU(negative_slope = leaky, inplace=True)
	elif nl=='HS':
		nlin_layer = HSwish(inplace=True)

	return nn.Sequential(
		nn.Conv2d(inp, oup, 3, s, 1, bias=False),
		nn.BatchNorm2d(oup),
		nlin_layer,
	)

def conv_bn_no_nlin(inp, oup, s=1):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 3, s, 1, bias=False),
		nn.BatchNorm2d(oup),
	)

def conv_1x1_bn(inp, oup, s=1, nl='RE', leaky=0):
	if nl=='RE':
		nlin_layer = nn.LeakyReLU(negative_slope = leaky, inplace=True)
	elif nl=='HS':
		nlin_layer = HSwish(inplace=True)

	return nn.Sequential(
		nn.Conv2d(inp, oup, 1, s, 0, bias=False),
		nn.BatchNorm2d(oup),
		# nn.LeakyReLU(negative_slope=leaky, inplace=True)
		nlin_layer,
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

class SEBlock(nn.Module):
	def __init__(self, channel, reduction=4):
		super(SEBlock, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
				nn.Linear(channel, channel//reduction, bias=False),
				nn.LeakyReLU(negative_slope=0.1, inplace=True),
				nn.Linear(channel//reduction, channel, bias=False),
				HSwish(inplace=False),
			)
	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b,c)
		y = self.fc(y).view(b,c,1,1)
		return x*y.expand_as(x)

class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x


class BottleneckBlock(nn.Module):
	def __init__(self, inp, oup, k, s, exp, se=False, nl='RE'):
		super(BottleneckBlock, self).__init__()
		assert k in [3,5]
		assert s in [1,2]
		p = (k-1)//2
		self.use_res_block = s==1 and inp==oup

		if nl=='RE':
			nlin_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
		elif nl=='HS':
			nlin_layer = HSwish(inplace=True)
		else:
			raise NotImplementedError

		if se:
			SELayer = SEBlock(exp)
		else:
			SELayer = Identity()

		self.conv = nn.Sequential(
				# pw
				nn.Conv2d(inp, exp, 1, 1, 0, bias=False),
				nn.BatchNorm2d(exp),
				nlin_layer,
				#dw
				nn.Conv2d(exp, exp, k, s, p, bias=False),
				nn.BatchNorm2d(exp),
				SELayer,
				nlin_layer,
				# pw-linear
				nn.Conv2d(exp, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup),
			)

	def forward(self, x):
		if self.use_res_block:
			return x + self.conv(x)
		else:
			return self.conv(x)

def make_divisible(x, divisible_by=8):
	return int(np.ceil(x*1./divisible_by)*divisible_by)

class SSH(nn.Module):
	def __init__(self, inp, oup):
		super(SSH, self).__init__()
		self.leaky = 0
		if (oup <= 64):
			self.leaky = 0.1
		self.conv3x3 = conv_bn_no_nlin(inp, oup//2, s=1)

		self.conv5x5_1 = conv_bn(inp, oup//4, s=1, nl='RE', leaky=self.leaky)
		self.conv5x5 = conv_bn_no_nlin(oup//4, oup//4, s=1)

		self.conv7x7_1 = conv_bn(oup//4, oup//4, s=1, nl='RE', leaky=self.leaky)
		self.conv7x7 = conv_bn_no_nlin(oup//4, oup//4, s=1)

	def forward(self, x):
		y3x3 = self.conv3x3(x)

		y5x5_1 = self.conv5x5_1(x)
		y5x5 = self.conv5x5(y5x5_1)

		y7x7_1 = self.conv7x7_1(y5x5_1)
		y7x7 = self.conv7x7(y7x7_1)

		out = torch.cat([y3x3, y5x5, y7x7], dim=1)
		out = F.leaky_relu(out, negative_slope=self.leaky, inplace=False)
		return out

class FPN(nn.Module):
	def __init__(self, inps, oup):
		super(FPN, self).__init__()
		leaky = 0
		if (oup <= 64):
			leaky = 0.1
		self.out1 = conv_1x1_bn(inps[0], oup, s=1, nl='RE', leaky=leaky)
		self.out2 = conv_1x1_bn(inps[1], oup, s=1, nl='RE', leaky=leaky)
		self.out3 = conv_1x1_bn(inps[2], oup, s=1, nl='RE', leaky=leaky)

		self.merge1 = conv_bn(oup, oup, s=1, nl='RE', leaky=leaky)
		self.merge2 = conv_bn(oup, oup, s=1, nl='RE', leaky=leaky)

	def forward(self, features):
		features = list(features.values())

		out1 = self.out1(features[0])
		out2 = self.out2(features[1])
		out3 = self.out3(features[2])

		# up3 = F.interpolate(out3, scale_factor=2, mode='nearest')
		_, _, H, W = out2.size()
		up3 = F.interpolate(out3, size=(H,W), mode='nearest')
		# print(out3.shape)
		# print(out2.shape)
		# print(out1.shape)
		out2 = out2 + up3
		out2 = self.merge2(out2)

		# up2 = F.interpolate(out2, scale_factor=2, mode='nearest')
		_, _, H, W = out1.size()
		up2 = F.interpolate(out2, size=(H,W), mode='nearest')
		out1 = out1 + up2
		out1 = self.merge1(out1)

		out = [out1, out2, out3]
		return out

class MobileNetV3(nn.Module):
	def __init__(self, n_cls = 1000, input_sz=224, dropout=0.8, mode='small', width_mult=1.0):
		super(MobileNetV3, self).__init__()
		input_channel = 16 
		last_channel = 1280
		option = mode

		if option == 'large':
			setting = [
			   # k, exp, c,   se,    nl,  s
				[[3, 16,  16, False, 'RE', 1],
				[3, 64,  24, False, 'RE', 2],
				[3, 72,  24, False, 'RE', 1],
				[5, 72,  40, True,  'RE', 2],
				[5, 120, 40, True,  'RE', 1],
				[5, 120, 40, True,  'RE', 1],],
				[[3, 240, 80, False, 'HS', 2],
				[3, 200, 80, False, 'HS', 1],
				[3, 184, 80, False, 'HS', 1],
				[3, 184, 80, False, 'HS', 1],
				[3, 480, 112, True, 'HS', 1],
				[3, 672, 112, True, 'HS', 1],],
				[[3, 672, 160, True, 'HS', 2],
				[3, 960, 160, True, 'HS', 1],
				[3, 960, 160, True, 'HS', 1],]
			]
		elif option == 'small':
			pass

		else:
			raise NotImplementedError


		#build first layer
		assert input_channel % 16 == 0
		last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
		self.classifier = []
		self.stage1 = [conv_bn(3, input_channel, 2, nl='HS')]
		self.stage2 = []
		self.stage3 = []
		self.last_stage = []

		for i, stage in enumerate(setting):
			for k, exp, c, se, nl, s in stage:
				oup_channel = make_divisible(c * width_mult)
				exp_channel = make_divisible(exp * width_mult)
				if i == 0:
					self.stage1.append(BottleneckBlock(input_channel, oup_channel, k, s, exp_channel, se, nl))
				elif i == 1:
					self.stage2.append(BottleneckBlock(input_channel, oup_channel, k, s, exp_channel, se, nl))
				elif i == 2:
					self.stage3.append(BottleneckBlock(input_channel, oup_channel, k, s, exp_channel, se, nl))
				input_channel = oup_channel

		last_conv = make_divisible(960 * width_mult)
		self.last_stage.append(conv_1x1_bn(input_channel, last_conv, s=1, nl='HS', leaky=0))
		self.last_stage.append(nn.AdaptiveAvgPool2d(1))
		self.last_stage.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
		self.last_stage.append(HSwish(inplace=True))

		self.stage1 = nn.Sequential(*self.stage1)
		self.stage2 = nn.Sequential(*self.stage2)
		self.stage3 = nn.Sequential(*self.stage3)	

		self.last_stage = nn.Sequential(*self.last_stage)

		self.classifier = nn.Sequential(
				nn.Dropout(p=dropout),
				nn.Linear(last_channel, n_cls),
			)

	def forward(self, x):
		x = self.stage1(x)
		x = self.stage2(x)
		x = self.stage3(x)
		x = self.last_stage(x)
		x = x.mean(3).mean(2)
		x = self.classifier(x)
		return x

if __name__=="__main__":
	backbone = MobileNetV3(n_cls=1000, input_sz=224, dropout=0.8, mode='large', width_mult=1.0)
	print(backbone)
	img = np.random.randint(0, 255, (3,224,224)).astype(np.float32)
	img = torch.from_numpy(img)
	img = img.unsqueeze(0)
	print(img.shape)
	y = backbone(img)
	print(y)
	print(y.shape)

