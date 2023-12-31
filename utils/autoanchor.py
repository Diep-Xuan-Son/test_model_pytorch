import torch
from itertools import product as product
import numpy as np
from math import ceil

class PriorBox(object):
	def __init__(self, image_size=None, phase='train'):
		super(PriorBox, self).__init__()
		self.min_sizes = [[[10, 13], [16, 30], [33, 23]], \
						[[30, 61], [62, 45], [59, 119]], \
						[[116, 90], [156, 198], [373, 326]]]
		# self.min_sizes = [[16, 32], [64, 128], [256, 512]]
		self.steps = [8, 16, 32]
		self.clip = False
		self.image_size = [image_size, image_size]
		self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
		self.name = "s"

	def forward(self):
		anchors = []
		for k, f in enumerate(self.feature_maps):
			min_sizes = self.min_sizes[k]
			for i, j in product(range(f[0]), range(f[1])):
				for min_size in min_sizes:
					s_kx = min_size[0] / self.image_size[1]
					s_ky = min_size[1] / self.image_size[0]
					dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
					dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
					for cy, cx in product(dense_cy, dense_cx):
						anchors += [cx, cy, s_kx, s_ky]

		# back to torch land
		output = torch.Tensor(anchors).view(-1, 4)
		if self.clip:
			output.clamp_(max=1, min=0)
		return output

if __name__=="__main__":
	priorbox = PriorBox(image_size=640, phase='train')
	priors = priorbox.forward()
	print(priors)
	print(priors.shape)