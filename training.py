import gpytorch
from gpytorch.models import ExactGP, IndependentModelList
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.kernels import ScaleKernel, MultitaskKernel
from gpytorch.kernels import RBFKernel, RBFKernel, ProductKernel
from gpytorch.likelihoods import GaussianLikelihood, LikelihoodList, MultitaskGaussianLikelihood
from gpytorch.mlls import SumMarginalLogLikelihood, ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal

# PyTorch
import torch

# Math, avoiding memory leak, and timing
import math
import gc
import math
from Models import * 

def train_gp_batched_scalar(dataloader, use_cuda=False, epochs=10,
                            lr=0.1, thr=0, use_ard=False, composite_kernel=False,
                            ds=None, global_hyperparams=False,
                            model_hyperparams=None):
    """Computes a Gaussian Process object using GPyTorch. Each outcome is
    modeled as a single scalar outcome.
    Parameters:
        Zs (np.array): Array of inputs of expanded shape (B, N, XD), where B is
            the size of the minibatch, N is the number of data points in each
            GP (the number of neighbors we consider in IER), and XD is the
            dimensionality of the state-action space of the environment.
        Ys (np.array): Array of predicted values of shape (B, N, YD), where B is the
            size of the minibatch and N is the number of data points in each
            GP (the number of neighbors we consider in IER), and YD is the
            dimensionality of the state-reward space of the environment.
        use_cuda (bool): Whether to use CUDA for GPU acceleration with PyTorch
            during the optimization step.  Defaults to False.
        epochs (int):  The number of epochs to train the batched GPs over.
            Defaults to 10.
        lr (float):  The learning rate to use for the Adam optimizer to train
            the batched GPs.
        thr (float):  The mll threshold at which to stop training.  Defaults to 0.
        use_ard (bool):  Whether to use Automatic Relevance Determination (ARD)
            for the lengthscale parameter, i.e. a weighting for each input dimension.
            Defaults to False.
        composite_kernel (bool):  Whether to use a composite kernel that computes
            the product between states and actions to compute the variance of y.
        ds (int): If using a composite kernel, ds specifies the dimensionality of
            the state.  Only applicable if composite_kernel is True.
        global_hyperparams (bool):  Whether to use a single set of hyperparameters
            over an entire model.  Defaults to False.
        model_hyperparams (dict):  A dictionary of hyperparameters to use for
            initializing a model.  Defaults to None.
    Returns:
        model (BatchedGP): A GPR model of BatchedGP type with which to generate
            synthetic predictions of rewards and next states.
        likelihood (GaussianLikelihood): A likelihood object used for training
            and predicting samples with the BatchedGP model.
    """
    # Preprocess batch data
    train_x, train_y, test_x, test_y = dataloader.__getitem__(0)
    batch_shape = len(dataloader)

    if use_cuda:  # If GPU available
        output_device = torch.device('cuda:0')  # GPU

    # initialize likelihood and model
    likelihood = GaussianLikelihood(batch_shape=torch.Size([batch_shape]))

    # Determine which type of kernel to use
    if composite_kernel:
        model = CompositeBatchedGP(train_x, train_y, likelihood, batch_shape,
                          output_device, use_ard=use_ard, ds=ds)
    else:
        model = BatchedGP(train_x, train_y, likelihood, batch_shape,
                          output_device, use_ard=use_ard)

    # Initialize the model with hyperparameters
    if model_hyperparams is not None:
        model.initialize(**model_hyperparams)

    # Determine if we need to optimize hyperparameters
    if global_hyperparams:
        if use_cuda:  # Send everything to GPU for training
            model = model.cuda().eval()

            # Empty the cache from GPU
            torch.cuda.empty_cache()
            gc.collect()  # NOTE: Critical to avoid GPU leak
            del train_x, train_y, Zs, Ys, likelihood

        return model, model.likelihood

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    if use_cuda:  # Send everything to GPU for training
        model = model.cuda()
        likelihood = likelihood.cuda()
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        mll = mll.cuda()

    def epoch_train(j):
        """Helper function for running training in the optimization loop.  Note
        that the model and likelihood are updated outside of this function as well.
        Parameters:
            j (int):  The epoch number.
        Returns:
            item_loss (float):  The numeric representation (detached from the
                computation graph) of the loss from the jth epoch.
        """
        optimizer.zero_grad()  # Zero gradients
        output = model(train_x)  # Forwardpass
        loss = -mll(output, train_y).sum()  # Compute ind. losses + aggregate
        loss.backward()  # Backpropagate gradients
        item_loss = loss.item()  # Extract loss (detached from comp. graph)
        optimizer.step()  # Update weights
        optimizer.zero_grad()  # Zero gradients
        gc.collect()  # NOTE: Critical to avoid GPU leak
        return item_loss

    # Run the optimization loop
    for i in range(epochs):
        loss_i = epoch_train(i)
        if i % 10 == 0:
            print("LOSS EPOCH {}: {}".format(i, loss_i))
        if loss_i < thr:  # If we reach a certain loss threshold, stop training
            break

    # Empty the cache from GPU
    torch.cuda.empty_cache()

    return model, likelihood