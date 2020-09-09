import numpy as np 
from PIL import Image 
import cv2
import matplotlib.pyplot as plt 


def image_bbox(image_path):
	"""
	returns array
	each element in array
	label, x_min, x_max, y_min, y_max
	"""
	A = Image.open(image_path)
	A = np.array(A)
	contour_stack = []
	for i in np.unique(A):
		if i >= 2:
			#create a binary mask for that label
			B = np.zeros(A.shape, dtype=np.uint8)
			B[A==i] = 1
			_, contours, _ = cv2.findContours(B, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			for c in contours:
				x, y, w, h = cv2.boundingRect(c)
				contour_stack.append([i, x, x + w, y, y + h])
	return contour_stack
				