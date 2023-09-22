import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if ROOT not in sys.path:
	sys.path.append(ROOT)

from models.blocks import MobileNetV3, FPN, SSH
import numpy as np
import torch
import torch.nn as nn
import torchvision.models._utils as _utils

class ClassHead(nn.Module):
	def __init__(self, inp=512, num_anchor=3, num_cls=2):
		super(ClassHead, self).__init__()
		self.num_cls = num_cls
		self.num_anchor = num_anchor
		self.conv1x1 = nn.Conv2d(inp, num_anchor*self.num_cls, 1, 1, 0)

	def forward(self, x):
		out = self.conv1x1(x)
		out = out.permute(0,2,3,1).contiguous()
		return out.view(-1, int(out.size(1)*out.size(2)*self.num_anchor), self.num_cls)

class BboxHead(nn.Module):
	def __init__(self, inp=512, num_anchor=3):
		super(BboxHead, self).__init__()
		self.num_anchor = num_anchor
		self.conv1x1 = nn.Conv2d(inp, num_anchor*4, 1, 1, 0)

	def forward(self, x):
		out = self.conv1x1(x)
		out = out.permute(0, 2, 3, 1).contiguous()
		return out.view(-1, int(out.size(1)*out.size(2)*self.num_anchor), 4)

class LandmarkHead(nn.Module):
	def __init__(self, inp=512, num_anchor=3):
		super(LandmarkHead, self).__init__()
		self.num_anchor = num_anchor
		self.conv1x1 = nn.Conv2d(inp, num_anchor*10, 1, 1, 0)

	def forward(self, x):
		out = self.conv1x1(x)
		out = out.permute(0, 2, 3, 1).contiguous()
		return out.view(-1, int(out.size(1)*out.size(2)*self.num_anchor), 10)

class MyNet(nn.Module):
	def __init__(self, phase='train', num_cls=2):
		super(MyNet, self).__init__()
		self.phase = phase
		self.num_cls = num_cls

		pretrained = False
		backbone = None 
		backbone = MobileNetV3(n_cls = 1000, input_sz=224, dropout=0.8, mode='large', width_mult=1.0)
		if pretrained:
			checkpoint = torch.load("./weights/mobilenetv3_small_67.4.pth.tar", map_location=torch.device("cpu"))
			# print(checkpoint.keys())
			# from collections import OrderedDict
            #     new_state_dict = OrderedDict()
            #     for k, v in checkpoint['state_dict'].items():
            #         name = k[7:]  # remove module.
            #         new_state_dict[name] = v
			backbone.load_state_dict(checkpoint)
		self.body = _utils.IntermediateLayerGetter(backbone, return_layers={"stage1": "st1", "stage2": "st2", "stage3": "st3"})
		
		self.fpn = FPN([40, 112, 160], 64)
		self.ssh1 = SSH(64, 64)
		self.ssh2 = SSH(64, 64)
		self.ssh3 = SSH(64, 64)

		self.classhead = self._make_class_head(fpn_num=3, inp=64, num_anchor=2)
		self.bboxhead = self._make_bbox_head(fpn_num=3, inp=64, num_anchor=2)
		self.landmarkhead = self._make_landmark_head(fpn_num=3, inp=64, num_anchor=2)

	def _make_class_head(self, fpn_num=3, inp=64, num_anchor=2):
		classhead = nn.ModuleList()
		for i in range(fpn_num):
			classhead.append(ClassHead(inp=inp, num_anchor=num_anchor, num_cls=self.num_cls))
		return classhead

	def _make_bbox_head(self, fpn_num=3, inp=64, num_anchor=2):
		bboxhead = nn.ModuleList()
		for i in range(fpn_num):
			bboxhead.append(BboxHead(inp=inp, num_anchor=num_anchor))
		return bboxhead

	def _make_landmark_head(self, fpn_num=3, inp=64, num_anchor=2):
		landmarkhead = nn.ModuleList()
		for i in range(fpn_num):
			landmarkhead.append(LandmarkHead(inp=inp, num_anchor=num_anchor))
		return landmarkhead

	def forward(self, x):
		# print(self.body)
		out = self.body(x)

		fpn = self.fpn(out)
		# print(fpn)
		feature1 = self.ssh1(fpn[0])
		feature2 = self.ssh2(fpn[1])
		feature3 = self.ssh3(fpn[2])
		features = [feature1, feature2, feature3]

		class_regressions = torch.cat([self.classhead[i](ft) for i, ft in enumerate(features)], dim=1)
		bbox_regressions = torch.cat([self.bboxhead[i](ft) for i, ft in enumerate(features)], dim=1)
		landmark_regressions = torch.cat([self.landmarkhead[i](ft) for i, ft in enumerate(features)], dim=1)

		return (class_regressions, bbox_regressions, landmark_regressions)

if __name__=="__main__":
	mynet = MyNet()
	img = np.random.randn(3,480,480).astype(np.float32)
	img = np.expand_dims(img, 0)
	img = torch.from_numpy(img)

	out = mynet(img)
	print(out[1].shape)