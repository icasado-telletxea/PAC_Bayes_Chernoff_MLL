
import torch.nn as nn
import torch
from torchvision import datasets, transforms
import numpy as np
import torchvision
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from laplace import Laplace
from tqdm import tqdm

from utils import latex_format, eval, eval_laplace

# Activate Latex format for matplotlib
latex_format()

# Create custom loss functions
criterion = nn.CrossEntropyLoss() # supervised classification loss

# Hyper-Parameters
RANDOM_SEED = 2147483647
LEARNING_RATE = 0.01
SUBSET_SIZE = 50000
TEST_SUBSET_SIZE = 10000
N_ITERS = 2000000
BATCH_SIZE = 200
BATCH_SIZE_TEST = 1000
IMG_SIZE = 32
N_CLASSES = 10
WEIGHT_DECAY = 0.01


# setup devices
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.manual_seed(RANDOM_SEED)
else:
    device = torch.device("cpu")

#######################################################################
############################# DATASET #################################
#######################################################################

transforms = torchvision.transforms.Compose([
                  torchvision.transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


train_dataset = datasets.CIFAR10(root='cifar_data',
                                train=True,
                                transform=transforms,
                                download=True)

test_dataset = datasets.CIFAR10(root='cifar_data',
                                train=False,
                                transform=transforms)

train_dataset = torch.utils.data.Subset(train_dataset, list(range(0, SUBSET_SIZE)))
test_dataset = torch.utils.data.Subset(test_dataset, list(range(0, TEST_SUBSET_SIZE)))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                          batch_size=BATCH_SIZE_TEST,
                          shuffle=False)


#######################################################################
############################# TRAIN MODELS ############################
#######################################################################



labels = np.loadtxt("models/model_labels.txt", delimiter=" ", dtype = str)
n_params = np.loadtxt("models/n_params.txt")

log_marginal = []
Gibbs_losses = []
Bayes_losses = []
prior_precisions = []
subset = "last_layer"
hessian = "kron"
with tqdm(range(len(labels))) as t:
  for i in range(len(labels)):

    with open(f"models/{labels[i]}.pickle", "rb") as handle:
      model = pickle.load(handle)
      la = Laplace(model, "classification",
                  subset_of_weights=subset,
                  hessian_structure=hessian)
      la.load_state_dict(torch.load(f'laplace_models/{labels[i]}_{subset}_{hessian}_state_dict.pt'))

      log_marginal.append(la.log_marginal_likelihood(la.prior_precision).detach().cpu().numpy()) 
      bayes_loss, gibbs_loss = eval_laplace(device, la, test_loader)
      Bayes_losses.append(bayes_loss.detach().cpu().numpy())
      Gibbs_losses.append(gibbs_loss.detach().cpu().numpy())
      prior_precisions.append(la.prior_precision.detach().cpu().numpy().item())
      t.set_description(f"Model {labels[i]}")
      t.update(1)

map_results = pd.read_csv("results/train_results.csv")
results = pd.DataFrame({'model': labels, 'parameters': n_params, 
                        'subset': subset, 'hessian': hessian, 
                        "prior precision": prior_precisions, 
                       "Bayes loss": Bayes_losses, 
                       "Gibbs loss": Gibbs_losses, 
                       "log marginal": log_marginal})
results.to_csv("results/laplace_results.csv", index=False)
print(results)