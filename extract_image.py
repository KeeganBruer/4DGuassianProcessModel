from Models import GPModel

import glob
import torch
import gpytorch

from random import randint
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
import numpy as np
import os, sys
import math
from TrainingDataset import TrainingData
from Dataset_converter import convert_dataset

import plotly.express as px
import plotly.graph_objects as go
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageExtractor():
    def __init__(self, config):
        self.img_size = config["img_size"]
        batch_size = config["batch_size"]
        r_dim = config["r_dim"]
        h_dim = config["h_dim"]
        z_dim = config["z_dim"]
        self.num_context_range = config["num_context_range"]
        self.num_extra_target_range = config["num_extra_target_range"]
        epochs = config["epochs"]
        self.save_directory = config["save_directory"]
        self.data_directory = config["data_directory"]
        
        dataloader = TrainingData(num_samples=200, points_per_file=10000, max_points=3000, path_to_data=config["data_directory"], device=device)
        self.dataloader = dataloader

        train_x, train_y, test_x, test_y = dataloader.__getitem__(0)
        batch_shape = len(train_x)
        train_x = train_x[0]
        train_y = train_y[0]
        inducing_points = train_x[:400, :]
        self.model = GPModel(inducing_points=inducing_points)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
        self.model.load_state_dict(torch.load(self.save_directory + '/model.pt', map_location=lambda storage, loc: storage))
        self.likelihood.load_state_dict(torch.load(self.save_directory + '/likelihood.pt', map_location=lambda storage, loc: storage))

        self.model.eval()
        self.likelihood.eval()


        if not os.path.exists("images"):
            os.mkdir("images")

    def extract_images(self):
        width, height, time = 10, 10, 0.00
        view_distance = 10
        origin = [0, -3, 2]
        world_q = [-0.16864013, 0.18102272, 0.66044826, 0.70894243]
        points = convert_dataset(origin, width, height, view_distance, world_q)
        for i, data in enumerate(self.dataloader):
            x, y, _, _ = data  # data is a tuple (img, label)
            x_context = x.to(device)
            y_context = y.to(device)
            graph_points2 = []
            new_x = x_context.cpu().numpy()
            new_y = y_context.cpu().numpy()
            times_ = []
            print("context points {}".format(len(new_x[0])))
            for i in range(len(new_x[0])):
                point1 = new_x[0][i][0:3]
                p_time = new_x[0][i][3]
                point2 = new_x[0][i][4:7]
                distance = new_y[0][i][0]
                if (not p_time in times_):
                    idx = len(times_)
                    times_.append(p_time)
                    graph_points2.append([])
                else:
                    idx = times_.index(p_time)
                #print("point {} point {} p_time {} distance {}".format(point1, point2, p_time, distance))
                midpoint = [(point1[0]+point2[0])* distance, (point1[1]+point2[1]) * distance, point1[2]+point2[2] * distance]
                graph_points2[idx].append(point1)
                graph_points2[idx].append(midpoint)
            print("context times {}".format(times_))
            time = times_[0] + ((times_[1]-times_[0])/2)
            x_target = []
            
            for point in points:
                x_target.append([*origin, time, *point, time])
            x_target = [x_target]
            print(np.array(x_context.to(torch.device("cpu"))).shape)
            print(np.array(x_target).shape)
            x_target = torch.Tensor(np.array(x_target)).to(device)
            
            p_y_pred = self.model(x_target)
            mean = p_y_pred.mean
            print("mean")
            print(mean)
            p_y_pred = self.likelihood(p_y_pred)
            x_target = x_target.cpu().numpy()
            y_target = p_y_pred.loc.detach().cpu().numpy()
            print(y_target)
            continue
            graph_points = []
            for i in range(len(x_target[0])):
                point1 = x_target[0][i][0:3]
                p_time = x_target[0][i][3]
                point2 = x_target[0][i][4:7]
                distance = y_target[0][i][0]
                #print("point {} point {} p_time {} distance {}".format(point1, point2, p_time, distance))
                midpoint = [(point1[0]+point2[0])* distance, (point1[1]+point2[1]) * distance, point1[2]+point2[2] * distance]
                graph_points.append(point1)
                graph_points.append(midpoint)
            
            #print(len(graph_points))
            
            f = open("display_points/result_points_{0:0.4f}.txt".format(time), "w")
            f.write("!@!title: Time: {0:0.4f}\n".format(time))
            f.write("!@!name: model points\n")
            for point in graph_points:
                f.write("{}, {}, {}\n".format(*point))  
            for i, points2 in enumerate(graph_points2):
                f.write("new trace\n")
                f.write("!@!name: context points {}\n".format(times_[i]))
                for point in points2:
                    f.write("{}, {}, {}\n".format(*point))
            f.write("new trace\n")
            f.write("!@!name: reference points\n")  
            for point in points:
                f.write("{}, {}, {}\n".format(*point))
            f.close()
            time += 0.01
            continue
            fig = go.Figure()
            #fig.update_xaxes(range=[140, 190])
            #fig.update_traces(mode='markers')
            #fig.add_trace(go.Scatter3d(
            #    x=[point[0] for point in points],
            #    y=[point[1] for point in points],
            #    z=[point[2] for point in points],
            #    mode='markers',
            #))
            fig.add_trace(go.Scatter3d(
                x=[point[0] for point in graph_points],
                y=[point[1] for point in graph_points],
                z=[point[2] for point in graph_points],
                mode='markers',
            ))
            fig.add_trace(go.Scatter3d(
                x=[point[0] for point in graph_points2],
                y=[point[1] for point in graph_points2],
                z=[point[2] for point in graph_points2],
                mode='markers',
            ))
            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-1.5, y=-1.5, z=.75)
            )

            fig.update_layout(scene_camera=camera, title="time: {0:0.3f}".format(time))
            #fig.write_image(os.getcwd()+"/images/frame_{0:0.3f}.png".format(time))
            fig.show()
            time += 0.1

if __name__ == "__main__":
    config_path = ""
    if (len(sys.argv) > 1):
        config_path = sys.argv[1]
    config = {}
    with open(config_path if config_path != "" else "./testing_config.json") as config_file:
        config = json.load(config_file)
    ImgExtractor = ImageExtractor(config)
    ImgExtractor.extract_images()
