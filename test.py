import torch
import gpytorch
import json


from TrainingDataset import TrainingData
from Models import GPModel

config = {
    "data_directory":"./training_data/",
    "save_directory": "./results",
    "epochs": 100,
}

if torch.cuda.is_available():  # If GPU available
    output_device = torch.device('cuda:0')  # GPU
else:
    output_device = torch.device('cpu:0')


dataloader = TrainingData(num_samples=200, points_per_file=10000, max_points=3000, path_to_data=config["data_directory"], device=output_device)

train_x, train_y, test_x, test_y = dataloader.__getitem__(0)
batch_shape = len(train_x)
train_x = train_x[0]
train_y = train_y[0]
inducing_points = train_x[:400, :]
model = GPModel(inducing_points=inducing_points)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()


model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.01)

# Our loss object. We're using the VariationalELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))


losses = []
for i in range(config["epochs"]):
    # Within each iteration, we will go over each minibatch of data
    avg_loss = 0
    for j in range(len(dataloader)):
        train_x, train_y, test_x, test_y = dataloader.__getitem__(j)
        train_x = train_x[0] #get the single batch
        train_y = train_y[0]
        
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        mean_ = torch.mean(loss)
        #mean_.grad_fn = loss.grad_fn
        #loss_val = loss.item().numpy()
        print(mean_.item())
        avg_loss += mean_.item()
        mean_.backward()
        optimizer.step()
    avg_loss = avg_loss/len(dataloader)
    losses.append(avg_loss)
    with open(config["save_directory"] + '/losses.json', 'w') as f:
        json.dump(losses, f, indent=4)
    torch.save(model.state_dict(), config["save_directory"] + '/model.pt')
    torch.cuda.empty_cache()
print(losses)