import torch
import cv2
from torch.utils import data
import torch.optim as optim

from data.argument import Preproc
from data.mydata import MyDataset, detection_collate
from models.mynet import MyNet
from utils.autoanchor import PriorBox
from utils.loss import MultiBoxLoss

def train(label_path, img_size, rgb_means, epoch, device):
	preproc = Preproc(img_size, rgb_means)
	mydata = MyDataset(label_path, preproc)

	train_dataloader = data.DataLoader(mydata, batch_size=2, shuffle=True, num_workers=2, collate_fn=detection_collate)

	# iterator = iter(train_dataloader)

	# batch = next(iterator)
	for ep in range(epoch):
		for i, (images, targets) in enumerate(train_dataloader):
			print(i)
			print(images.shape)
			# print(targets)
			if device in ['0', '1', '2'] and torch.cuda.is_available():
				images = images.cuda()
				targets = [anno.cuda() for anno in targets]

			# forward
			out = net(images)
			print(out[1].shape)

			# backprop
			optimizer.zero_grad()
			criterion(out, priors, targets)
			exit()


if __name__=="__main__":
	label_path = "./datasets/water_ruler_v2/label.txt"
	img_size, rgb_means = (640, (104, 117, 123))
	epoch = 1
	device = 'cpu'

	num_classes = 2
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