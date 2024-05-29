
import torch.nn as nn
import torch
from torchvision import datasets, transforms
import numpy as np
import torchvision
import pickle

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from utils import latex_format, createmodel, train

# Activate Latex format for matplotlib
latex_format()

# Create custom loss functions
criterion = nn.CrossEntropyLoss() # supervised classification loss
criterion_nonreduced = nn.CrossEntropyLoss(reduce=False) # supervised classification loss

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

train_dataset = torch.utils.data.Subset(train_dataset,
                                        list(range(0, SUBSET_SIZE)))


#######################################################################
############################# TRAIN MODELS ############################
#######################################################################

# Initialize channels for models
ks = np.arange(0.2, 5, step = 0.1)
ks = ks[-3:]

# Initialice models
models = [createmodel(k, RANDOM_SEED, 10, 3).to(device) for k in ks]

n_params = []
for model in models:
  n = 0
  for parameter in model.parameters():
    n += parameter.flatten().size(0)
  n_params.append(n)


labels = ["ConvNN-"+str(p//1000)+"k" for p in n_params]

np.savetxt("models/model_labels.txt",labels, delimiter=" ", fmt="%s")
np.savetxt("models/n_params.txt",n_params)


for i in range(len(models)):
  g_cuda = torch.Generator(device='cpu')
  g_cuda.manual_seed(RANDOM_SEED)
  loader = torch.utils.data.DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          generator=g_cuda,
                          shuffle=True)
  train(models[i], loader, LEARNING_RATE, N_ITERS, device, criterion)
  
  with open(f'models/{labels[i]}.pickle', 'wb') as handle:
    pickle.dump(models[i], handle, protocol=pickle.HIGHEST_PROTOCOL)

