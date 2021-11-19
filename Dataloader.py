import numpy as np
import math
import torch

class TrainingData:
    def __init__(self, location, use_cuda):
        self.x_train = []
        self.y_train = []
        self.x_samples=np.load(location+'/sample_set_1.npz')

        self.x_train.append(self.x_samples["sample_set_x"])
        
        raw_y =  self.x_samples["sample_set_y"]

        self.y_train.append(raw_y)
        
        if use_cuda:  # If GPU available
            output_device = torch.device('cuda:0')  # GPU

        # Format the training features - tile and reshape
        self.x_train = torch.tensor(self.x_train, device=output_device)

        # Format the training labels - reshape
        self.y_train = torch.tensor(self.y_train, device=output_device)
        
        
        
        
    def __getitem__(self, i):
        x, y =  self.x_train[i], self.y_train[i]
        return x[0:200], y[0:200]

    def __len__(self):
        return len(self.x_train)
    def get_shape(self):
        return self.get_length()