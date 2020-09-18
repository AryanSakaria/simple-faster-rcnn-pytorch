import numpy as np 
from PIL import Image 
import sys
sys.path.remove(sys.path[1])
import cv2
# from tqdm import tqdm
import os
# import matplotlib.pyplot as plt 


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
				contour_stack.append([i, y, x, y + h, x + w])
	contour_stack = np.array(contour_stack, dtype = np.uint16)
	return contour_stack


if __name__ == '__main__':
	#generate for all images.
	data_path = "/scratch/aryansakaria/SmallObstacle/Small_Obstacle_Dataset"
	# img_list = []
	# save_list = []
	for split in os.listdir(data_path):
		split_path = os.path.join(data_path, split)
		# print(split_path)
		for seq in os.listdir(split_path):
			seq_path = os.path.join(split_path, seq)
			labels_path = os.path.join(seq_path, "labels")
			bbox = os.path.join(seq_path, "bbox")

			if not os.path.isdir(bbox):
				os.makedirs(bbox)
			# print(os.listdir(seq_path))
			labels = os.listdir(labels_path)
			for label in labels:
				img_path = os.path.join(labels_path, label)
				save_path = label[:-4]
				save_path = os.path.join(bbox, save_path + ".npy")
				# img_list.append(img_path)
				# save_list.append(save_path)
				bbox_data = image_bbox(img_path)
				bbox_data = bbox_data.astype(np.int16)
				np.save(save_path, bbox_data)
				print(save_path)
				# print(bbox_data)
				# print(save_path)
				# print(img_path)
				# np.save(save_path, bbox_data)
				# print(img_path)


