# GPyTorch Imports
import gpytorch
import torch
import time

from training import train_gp_batched_scalar
from TrainingDataset import TrainingData
def main():
    
    
    """Main tester function."""
    # Set parameters
    B = 256  # Number of batches
    N = 100  # Number of data points in each batch
    D = 3  # Dimension of X and Y data
    Ds = 1  # Dimensions for first factored kernel - only needed if factored kernel is used
    EPOCHS = 50  # Number of iterations to perform optimization
    THR = -1e5  # Training threshold (minimum)
    USE_CUDA = torch.cuda.is_available()  # Hardware acceleraton
    MEAN = 0  # Mean of data generated
    SCALE = 1  # Variance of data generated
    COMPOSITE_KERNEL = False  # Use a factored kernel
    USE_ARD = True  # Use Automatic Relevance Determination in kernel
    LR = 0.5  # Learning rate
    config = {
        "data_directory":"./training_data/"
    }
    # GPyTorch training
    start = time.time()
    if USE_CUDA:  # If GPU available
        output_device = torch.device('cuda:0')  # GPU
    else:
        output_device = torch.device('cpu:0')
    dataloader = TrainingData(num_samples=200, points_per_file=10000, max_points=3000, path_to_data=config["data_directory"], device=output_device)
    
    model, likelihood = train_gp_batched_scalar(dataloader,
                                                use_cuda=USE_CUDA,
                                                composite_kernel=COMPOSITE_KERNEL,
                                                epochs=EPOCHS, lr=LR, thr=THR, ds=Ds,
                                                use_ard=USE_ARD)
    end = time.time()
    print("TRAINING TIME: {}".format(end - start))
    model.eval()
    likelihood.eval()
    
    
    
    

if __name__ == "__main__":
    main()