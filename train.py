import os
import torch
import cv2
from torch.utils import data
import torch.optim as optim

from data.argument import Preproc
from data.mydata import MyDataset, detection_collate
from models.mynet import MyNet
from utils.autoanchor import PriorBox
from utils.loss import MultiBoxLoss
import datetime
import time

def train(label_path, img_size, rgb_means, epoch, device):
	save_folder = "weight_result/"
	if not os.path.exists(save_folder):
		os.mkdir(save_folder)

	preproc = Preproc(img_size, rgb_means)
	mydata = MyDataset(label_path, preproc)
	mydata.get_anchors()

	train_dataloader = data.DataLoader(mydata, batch_size=2, shuffle=True, num_workers=2, collate_fn=detection_collate)

	# iterator = iter(train_dataloader)

	# batch = next(iterator)
	for ep in range(epoch):
		for i, (images, targets) in enumerate(train_dataloader):
			load_t0 = time.time()
			# lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)
			# print(targets)
			if device in ['0', '1', '2'] and torch.cuda.is_available():
				images = images.cuda()
				targets = [anno.cuda() for anno in targets]

			# forward
			out = net(images)
			# print(out[0])
			# exit()

			# backprop
			optimizer.zero_grad()
			loss_l, loss_c, loss_landm = criterion(out, priors, targets)
			loss = 2.0 * loss_l + loss_c + loss_landm
			loss.backward()
			optimizer.step()
			load_t1 = time.time()
			batch_time = load_t1 - load_t0
			# eta = int(batch_time * (max_iter - iteration))
			print('Epoch:{}/{} || Batch:{}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || Batchtime: {:.4f} s'
				.format(ep+1, epoch, i+1, len(train_dataloader), loss_l.item(), loss_c.item(), loss_landm.item(), batch_time))

	torch.save(net, save_folder + "model" + '_test.pth')
	exit()

def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
	"""Sets the learning rate
	# Adapted from PyTorch Imagenet example:
	# https://github.com/pytorch/examples/blob/master/imagenet/main.py
	"""
	warmup_epoch = -1
	if epoch <= warmup_epoch:
		lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
	else:
		lr = initial_lr * (gamma ** (step_index))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr
	
if __name__=="__main__":
	label_path = "./datasets/water_ruler_v2/label.txt"
	img_size, rgb_means = (640, (104, 117, 123))
	epoch = 5
	device = 'cpu'

	num_classes = 3
	net = MyNet(phase='train', num_cls=num_classes)
	# para = net.parameters()

	initial_lr = 1e-3
	momentum = 0.9
	weight_decay = 5e-4
	optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
	criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

	img_dim = 640
	priorbox = PriorBox(image_size=img_dim)
	with torch.no_grad():
		priors = priorbox.forward()
		# priors = priors.cuda()

	train(label_path, img_size, rgb_means, epoch, device)