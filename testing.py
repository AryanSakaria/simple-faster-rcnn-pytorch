import os
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from torch.utils import data as data_

"""
bounding boxes
labels
"""

if __name__ == '__main__':
	dataset = Dataset(opt)
	print("load data")
	dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)

	subset_indices = [0]
	subset = data_.Subset(dataset, subset_indices)
	testloader_subset = data_.DataLoader(subset, batch_size=1, num_workers=0, shuffle=False)

	for i,(img, bbox_, label_, scale) in enumerate(testloader_subset):
		print(i)