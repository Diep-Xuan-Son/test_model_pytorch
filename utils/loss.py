import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if ROOT not in sys.path:
	sys.path.append(ROOT)

import torch.nn as nn
import torch
from utils.box_utils import match

class MultiBoxLoss(nn.Module):
	def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
		super(MultiBoxLoss, self).__init__()
		self.num_classes = num_classes
		self.threshold = overlap_thresh
		self.background_label = bkg_label
		self.encode_target = encode_target
		self.use_prior_for_matching = prior_for_matching
		self.do_neg_mining = neg_mining
		self.negpos_ratio = neg_pos
		self.neg_overlap = neg_overlap
		self.variance = [0.1, 0.2]

	def forward(self, predictions, priors, targets):
		cls_data, loc_data, landm_data = predictions
		priors = priors
		num = loc_data.size(0)
		num_priors = (priors.size(0))

		# match priors (default boxes) and ground truth boxes
		loc_t = torch.Tensor(num, num_priors, 4)
		landm_t = torch.Tensor(num, num_priors, 10)
		conf_t = torch.LongTensor(num, num_priors)
		for idx in range(num):
			truths = targets[idx][:, :4].data
			labels = targets[idx][:, -1].data
			landms = targets[idx][:, 4:14].data
			defaults = priors.data
			match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)

# if __name__=="__main__":
# 	criterion = MultiBoxLoss(2, 0.35, True, 0, True, 7, 0.35, False)