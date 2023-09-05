import torch
import cv2
from torch.utils import data

from data.argument import Preproc
from data.mydata import MyDataset, detection_collate
from models.net import MyNet

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
			print(targets.shape)
			if device in ['0', '1', '2'] and torch.cuda.is_available():
				images = images.cuda()
        		targets = [anno.cuda() for anno in targets]

			exit()


if __name__=="__main__":
	label_path = "./datasets/water_ruler_v2/label.txt"
	img_size, rgb_means = (640, (104, 117, 123))
	epoch = 1
	device = 'cpu'

	net = MyNet()

	train(label_path, img_size, rgb_means, epoch, device)