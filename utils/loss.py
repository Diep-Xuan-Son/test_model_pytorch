import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if ROOT not in sys.path:
	sys.path.append(ROOT)

import torch.nn as nn
import torch.nn.functional as F
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
		zeros = torch.tensor(0)
		for idx in range(num):
			truths = targets[idx][:, :4].data
			labels = targets[idx][:, -1].data
			landms = targets[idx][:, 4:14].data
			defaults = priors.data
			match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
		# if GPU:
		# 	loc_t = loc_t.cuda()
		# 	conf_t = conf_t.cuda()
		# 	landm_t = landm_t.cuda()
		# 	zeros = zeros.cuda()

		pos1 = conf_t > zeros
		num_pos_landm = pos1.long().sum(1, keepdim=True)
		N1 = max(num_pos_landm.data.sum().float(), 1)
		pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
		landm_p = landm_data[pos_idx1].view(-1, 10)
		landm_t = landm_t[pos_idx1].view(-1, 10)
		loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')

		pos = conf_t != zeros
		pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
		loc_p = loc_data[pos_idx].view(-1, 4)
		loc_t = loc_t[pos_idx].view(-1, 4)
		loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

		# Compute max conf across batch for hard negative mining
		batch_conf = cls_data.view(-1, self.num_classes)
		loss_c = F.cross_entropy(batch_conf, conf_t.view(-1), reduction="none")
		''' calculate cross entropy:
				prob = np.exp(logits)/np.sum(np.exp(logits), axis=1)
				cross_entropy = (Ln(prob[0]) + Ln(prob[1]) + ... + Ln(prob[n]))/n
		'''
		loss_c = loss_c.view(num, -1)
		_, loss_idx = loss_c.sort(1, descending=True)
		_, idx_rank = loss_idx.sort(1)
		num_pos = pos.long().sum(1, keepdim=True)
		num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
		neg = idx_rank < num_neg.expand_as(idx_rank)

		# Confidence Loss Including Positive and Negative Examples
		pos_idx = pos.unsqueeze(2).expand_as(cls_data)
		neg_idx = neg.unsqueeze(2).expand_as(cls_data)
		conf_p = cls_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
		targets_weighted = conf_t[(pos+neg).gt(0)]
		loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

		N = max(num_pos.data.sum().float(), 1)
		loss_l /= N
		loss_c /= N
		loss_landm /= N1

		return loss_l, loss_c, loss_landm
# if __name__=="__main__":
# 	criterion = MultiBoxLoss(2, 0.35, True, 0, True, 7, 0.35, False)