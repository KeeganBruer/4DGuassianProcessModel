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
class BatchedGP(ExactGP):
    """Class for creating batched Gaussian Process Regression models.  Ideal candidate if
    using GPU-based acceleration such as CUDA for training.
    Parameters:
        train_x (torch.tensor): The training features used for Gaussian Process
            Regression.  These features will take shape (B * YD, N, XD), where:
                (i) B is the batch dimension - minibatch size
                (ii) N is the number of data points per GPR - the neighbors considered
                (iii) XD is the dimension of the features (d_state + d_action)
                (iv) YD is the dimension of the labels (d_reward + d_state)
            The features of train_x are tiled YD times along the first dimension.
        train_y (torch.tensor): The training labels used for Gaussian Process
            Regression.  These features will take shape (B * YD, N), where:
                (i) B is the batch dimension - minibatch size
                (ii) N is the number of data points per GPR - the neighbors considered
                (iii) YD is the dimension of the labels (d_reward + d_state)
            The features of train_y are stacked.
        likelihood (gpytorch.likelihoods.GaussianLikelihood): A likelihood object
            used for training and predicting samples with the BatchedGP model.
        shape (int):  The batch shape used for creating this BatchedGP model.
            This corresponds to the number of samples we wish to interpolate.
        output_device (str):  The device on which the GPR will be trained on.
        use_ard (bool):  Whether to use Automatic Relevance Determination (ARD)
            for the lengthscale parameter, i.e. a weighting for each input dimension.
            Defaults to False.
    """
    def __init__(self, train_x, train_y, likelihood, shape, output_device, use_ard=False):

        # Run constructor of superclass
        super(BatchedGP, self).__init__(train_x, train_y, likelihood)

        # Determine if using ARD
        ard_num_dims = None
        if use_ard:
            ard_num_dims = train_x.shape[-1]

        # Create the mean and covariance modules
        self.shape = torch.Size([shape])
        self.mean_module = ConstantMean(batch_shape=self.shape)
        self.base_kernel = RBFKernel(batch_shape=self.shape,
                                        ard_num_dims=ard_num_dims)
        self.covar_module = ScaleKernel(self.base_kernel,
                                        batch_shape=self.shape,
                                        output_device=output_device)

    def forward(self, x):
        """Forward pass method for making predictions through the model.  The
        mean and covariance are each computed to produce a MV distribution.
        Parameters:
            x (torch.tensor): The tensor for which we predict a mean and
                covariance used the BatchedGP model.
        Returns:
            mv_normal (gpytorch.distributions.MultivariateNormal): A Multivariate
                Normal distribution with parameters for mean and covariance computed
                at x.
        """
        mean_x = self.mean_module(x)  # Compute the mean at x
        covar_x = self.covar_module(x)  # Compute the covariance at x
        return MultivariateNormal(mean_x, covar_x)



class CompositeBatchedGP(ExactGP):
    """Class for creating batched Gaussian Process Regression models.  Ideal candidate if
    using GPU-based acceleration such as CUDA for training.
    This kernel produces a composite kernel that multiplies actions times states,
    i.e. we have a different kernel for both the actions and states.  In turn,
    the composite kernel is then multiplied by a Scale kernel.
    Parameters:
        train_x (torch.tensor): The training features used for Gaussian Process
            Regression.  These features will take shape (B * YD, N, XD), where:
                (i) B is the batch dimension - minibatch size
                (ii) N is the number of data points per GPR - the neighbors considered
                (iii) XD is the dimension of the features (d_state + d_action)
                (iv) YD is the dimension of the labels (d_reward + d_state)
            The features of train_x are tiled YD times along the first dimension.
        train_y (torch.tensor): The training labels used for Gaussian Process
            Regression.  These features will take shape (B * YD, N), where:
                (i) B is the batch dimension - minibatch size
                (ii) N is the number of data points per GPR - the neighbors considered
                (iii) YD is the dimension of the labels (d_reward + d_state)
            The features of train_y are stacked.
        likelihood (gpytorch.likelihoods.GaussianLikelihood): A likelihood object
            used for training and predicting samples with the BatchedGP model.
        shape (int):  The batch shape used for creating this BatchedGP model.
            This corresponds to the number of samples we wish to interpolate.
        output_device (str):  The device on which the GPR will be trained on.
        use_ard (bool):  Whether to use Automatic Relevance Determination (ARD)
            for the lengthscale parameter, i.e. a weighting for each input dimension.
            Defaults to False.
        ds (int): If using a composite kernel, ds specifies the dimensionality of
            the state.  Only applicable if composite_kernel is True.
    """
    def __init__(self, train_x, train_y, likelihood, shape, output_device,
                 use_ard=False, ds=None):

        # Run constructor of superclass
        super(CompositeBatchedGP, self).__init__(train_x, train_y, likelihood)

        # Check if ds is None, and if not, set
        if ds is None:
            raise Exception("No dimension for state specified.  Please specify ds.")
        self.ds = ds

        # Set active dimensions
        state_dims = torch.tensor([i for i in range(0, ds)])
        action_dims = torch.tensor([i for i in range(ds, train_x.shape[-1])])

        # Determine if using ARD
        state_ard_num_dims = None
        action_ard_num_dims = None
        if use_ard:
            state_ard_num_dims = ds
            action_ard_num_dims = train_x.shape[-1] - ds

        # Create the mean and covariance modules
        self.shape = torch.Size([shape])

        # Construct mean module
        self.mean_module = ConstantMean(batch_shape=self.shape)

        # Construct state kernel
        self.state_base_kernel = RBFKernel(batch_shape=self.shape,
                                              active_dims=state_dims,
                                              ard_num_dims=state_ard_num_dims)

        # Construct action kernel
        self.action_base_kernel = RBFKernel(batch_shape=self.shape,
                                               active_dims=action_dims,
                                               ard_num_dims=action_ard_num_dims)

        # Construct composite kernel
        self.composite_kernel = self.state_base_kernel * self.action_base_kernel
        self.covar_module = ScaleKernel(self.composite_kernel,
                                        batch_shape=self.shape,
                                        output_device=output_device)


    def forward(self, x):
        """Forward pass method for making predictions through the model.  The
        mean and covariance are each computed to produce a MV distribution.
        Parameters:
            x (torch.tensor): The tensor for which we predict a mean and
                covariance used the BatchedGP model.
        Returns:
            mv_normal (gpytorch.distributions.MultivariateNormal): A Multivariate
                Normal distribution with parameters for mean and covariance computed
                at x.
        """
        # Compute mean and covariance in batched form
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)