import numpy as np
import random
import cv2

def _resize_subtract_mean(image, imgsz, rgb_means):
	interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
	interp_method = interp_methods[random.randrange(len(interp_methods))]
	image = cv2.resize(image, (imgsz, imgsz), interpolation=interp_method)
	image = image.astype(np.float32)
	image -= rgb_means
	return image.transpose(2,0,1)

class Preproc(object):
	def __init__(self, img_size, rgb_means):
		self.imgsz = img_size
		self.rgb_means = rgb_means

	def __call__(self, img, target):
		assert target.shape[0] > 0, "this image does not have gt"

		boxes = target[:, :4].copy()
		labels = target[:, -1].copy()
		landms = target[:, 4:-1].copy()

		w, h, _ = img.shape

		img_t = _resize_subtract_mean(img, self.imgsz, self.rgb_means)
		boxes[:, 0::2] /= w 
		boxes[:, 1::2] /= h

		landms[:, 0::2] /= w
		landms[:, 1::2] /= h 

		labels = np.expand_dims(labels, 1)
		target_t = np.hstack((boxes, landms, labels))
		
		return img_t, target_t