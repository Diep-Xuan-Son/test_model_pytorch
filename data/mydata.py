import numpy as np
from torch.utils import data
import cv2
import torch


class MyDataset(data.Dataset):
	def __init__(self, label_path, preproc=None):
		# self.label_path = label_path
		self.preproc = preproc
		self.words = []
		self.list_img = []

		isFirst = True
		labels = []
		f = open(label_path, 'r')
		lines = f.readlines()

		for line in lines:
			line = line.rstrip()
			if line.startswith("#"):
				if isFirst:
					isFirst = False
				else:
					labels_cp = labels.copy()
					self.words.append(labels_cp)
					labels.clear()
				name_img = line[2:]
				path_img = label_path.replace('label.txt', 'images/') + name_img
				self.list_img.append(path_img)
			else:
				line = line.split(' ')
				label = [float(x) for x in line]
				labels.append(label)

		self.words.append(labels)

	def get_anchors(self):
		target = []
		for i, w in enumerate(self.words):
			if i==0:
				target = w
			else:
				target += w 
		target = np.array(target)
		wh = target[:, 2:4]
		print(wh)

	def __len__(self):
		return len(self.list_img)

	def __getitem__(self, idx):
		img = cv2.imread(self.list_img[idx])
		labels = self.words[idx]

		annotations = np.zeros((0,15))
		for i, label in enumerate(labels):
			#---------------box--------------
			annotation = np.zeros((1,15))
			annotation[0,0] = label[0]
			annotation[0,1] = label[1]
			annotation[0,2] = label[0] + label[2]
			annotation[0,3] = label[1] + label[3]

			#--------------landmark---------
			annotation[0,4] = label[4]
			annotation[0,5] = label[5]
			annotation[0,6] = label[7]
			annotation[0,7] = label[8]
			annotation[0,8] = label[10]
			annotation[0,9] = label[11]
			annotation[0,10] = label[13]
			annotation[0,11] = label[14]
			annotation[0,12] = label[16]
			annotation[0,13] = label[17]

			#-------------label------------
			annotation[0,14] = int(label[19])

			annotations = np.append(annotations, annotation, axis=0)

		target = np.array(annotations)
		if self.preproc is not None:
			img, target = self.preproc(img, target)

		return torch.from_numpy(img), target

def detection_collate(batch):
	"""Custom collate fn for dealing with batches of images that have a different
	number of associated object annotations (bounding boxes).

	Arguments:
		batch: (tuple) A tuple of tensor images and lists of annotations

	Return:
		A tuple containing:
			1) (tensor) batch of images stacked on their 0 dim
			2) (list of tensors) annotations for a given image are stacked on 0 dim
	"""

	targets = []
	imgs = []
	for _, samlpe in enumerate(batch):
		for _, tup in enumerate(samlpe):
			if torch.is_tensor(tup):
				imgs.append(tup)
			elif isinstance(tup, type(np.empty(0))):
				annos = torch.from_numpy(tup).float()
				targets.append(annos)
	return (torch.stack(imgs, 0), targets)

if __name__=="__main__":
	from argument import Preproc
	
	label_path = "../datasets/water_ruler_v2/label.txt"
	img_size, rgb_means = (640, (104, 117, 123))
	preproc = Preproc(img_size, rgb_means)
	mydata = MyDataset(label_path, preproc)

	train_dataloader = data.DataLoader(mydata, batch_size=1, shuffle=True, num_workers=2, collate_fn=detection_collate)

	iterator = iter(train_dataloader)

	batch = next(iterator)

	print(batch)