from torch.utils.data import Dataset, DataLoader
import glob
import torch

import json
from PIL import Image
import numpy as np
import os
import random
class TrainingData(Dataset):
    """
    Dataset of functions f(x) = a * sin(x - b) where a and b are randomly
    sampled. The function is evaluated from -pi to pi.

    Parameters
    ----------

    num_samples : int
        Number of samples of the function contained in dataset.
    """

    def __init__(self, num_samples=10, points_per_file=10, path_to_data="", max_points = None, device=None):
        self.num_samples = num_samples
        self.points_per_file = points_per_file
        self.dataset_paths = glob.glob(path_to_data+"/*")
        print(self.dataset_paths)
        self.data_paths = []
        for path in self.dataset_paths:
        	self.data_paths.append(sorted(glob.glob(path + '/*.npz'), key=lambda x: int(os.path.basename(x).replace(".npz", "").split('_')[-1])))
        self.device = device
        #print(self.data_paths)
        start_frame = np.load(self.data_paths[0][0])
        print(len(start_frame["sample_set_x"]), end=" ", flush=True)
        start_frame = np.load(self.data_paths[1][0])
        print(len(start_frame["sample_set_x"]), end=" ", flush=True)
        self.max_points = max_points if max_points != None else len(self.data_paths) * self.points_per_file
        #print("Max Points {}".format(self.max_points))

    def __getitem__(self, index):
        x_context = []
        y_context = []
        x_target = []
        y_target = []

        print("Request for {}:".format(index))
        first_index_of_dataset = 0
        last_index_of_dataset = len(self.data_paths[0])-2
        data_paths_i = 0
        for i in range(len(self.data_paths)):
            if (index >= last_index_of_dataset):
            	data_paths_i = i+1
            	first_index_of_dataset = last_index_of_dataset
            	last_index_of_dataset += len(self.data_paths[i])
            else:
            	break
        index = index - first_index_of_dataset
        print("\tGetting Data From dataset {} at frame {}".format(self.dataset_paths[data_paths_i], index))

        start_frame = np.load(self.data_paths[data_paths_i][index])
        target_frame = np.load(self.data_paths[data_paths_i][index+1])
        end_frame = np.load(self.data_paths[data_paths_i][index+2])
        #print(len(start_frame["sample_set_x"]), flush=True)
        for i in range(self.num_samples):
            #print(i, end=" ", flush=True)
            ridx1 = round(random.random() * (len(start_frame["sample_set_x"])-1))
            ridx2 = round(random.random() * (len(target_frame["sample_set_x"])-1))
            ridx3 = round(random.random() * (len(end_frame["sample_set_x"])-1))
            x_context.append(start_frame["sample_set_x"][ridx1])
            y_context.append([start_frame["sample_set_y"][ridx1]])

            x_target.append(target_frame["sample_set_x"][ridx2])
            y_target.append([target_frame["sample_set_y"][ridx2]])

            x_context.append(end_frame["sample_set_x"][ridx3])
            y_context.append([end_frame["sample_set_y"][ridx3]])
        x = torch.Tensor(np.array(x_context)).to(self.device)
        y = torch.Tensor(np.array(y_context)).to(self.device)
        x_t = torch.Tensor(np.array(x_target)).to(self.device)
        y_t = torch.Tensor(np.array(y_target)).to(self.device)
        print("\tFinished Getting Data")
        return x, y, x_t, y_t

    def __len__(self):
    	length = 0
    	for data_paths in self.data_paths:
    	    length += len(data_paths)-2
    	print("length {}".format(length))
    	return length
        
        
