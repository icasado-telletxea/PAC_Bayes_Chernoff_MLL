
import torch.nn as nn
import torch
from torchvision import datasets, transforms
import numpy as np
import torchvision
import pickle
import matplotlib.pyplot as plt
import pandas as pd

from utils import latex_format, eval

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

models = []
for i in range(len(labels)):

  with open(f"models/{labels[i]}.pickle", "rb") as handle:
    models.append(pickle.load(handle))

train_loss = []
test_loss = []
for i in range(len(models)):

  train_loss.append(eval(device, models[i], train_loader, criterion)[2])
  test_loss.append(eval(device, models[i], test_loader, criterion)[2])

results = pd.DataFrame({'model': labels, 'parameters': n_params, 
                        'train loss': train_loss, 'test loss': test_loss})
results.to_csv("results/train_results.csv", index=False)


# Window size for smoothing
N = 7
train_loss_ = *([train_loss[0]]*(N//2)), *train_loss, *([train_loss[-1]]*(N//2))
test_loss_ = *([test_loss[0]]*(N//2)), *test_loss, *([test_loss[-1]]*(N//2))

train_loss_ = np.convolve(train_loss_, np.ones(N)/N, mode = "valid")
test_loss_ = np.convolve(test_loss_, np.ones(N)/N, mode = "valid")

plt.rcParams['figure.figsize'] = (16, 8)

jet = plt.colormaps["Set2"]
lw = 5
plt.plot(n_params, train_loss, linewidth = lw, color = jet(1), alpha = 0.2)
plt.plot(n_params, test_loss, linewidth = lw, color = jet(2), alpha = 0.2)
plt.plot(n_params, train_loss_,  linewidth =lw, color = jet(1))
plt.plot(n_params, test_loss_, linewidth = lw, color = jet(2))
plt.scatter(n_params, train_loss, marker = "*", label = "Train loss", color = jet(1), s = 100)
plt.scatter(n_params, test_loss, marker = "o", label = "Test loss", color = jet(2), s = 100)

max_test = np.argmax(test_loss_)
mask = np.isclose(test_loss_, test_loss_[max_test], 0.05)
min_region = np.where(mask == True)[0][0]
max_region = np.where(mask == True)[0][-1]

plt.annotate("Classical Regime", xy = (0, 3.5))
plt.annotate("Variance-Bias Tradeoff", xy = (10000, 3.2), fontsize = 20)

plt.annotate("Modern Regime",xy = (n_params[max_region]+200000, 3.5), xytext = (n_params[max_test]+300000, 3.5))
plt.annotate("Larger is Better", xy = (n_params[max_test]+350000, 3.2), fontsize = 20)

plt.annotate("Interpolation Threshhold",xy = (n_params[max_test], 0.6), xytext = (n_params[max_test]+100000, 0.5), arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate("Critical Region",xy = (n_params[max_test]+100000, 1.1), xytext = (n_params[max_test]+200000, 1), arrowprops=dict(color='tomato', shrink=0.05))


plt.annotate(r"\textbf{Train loss}",xy = (n_params[max_test]+500000, 0.2), xytext = (n_params[max_test]+700000, 0.2), color = jet(1))
plt.annotate(r"\textbf{Test loss}",xy = (n_params[max_test]+500000, 2.1), xytext = (n_params[max_test]+700000, 2.1), color = jet(2))

plt.axvspan(n_params[min_region], n_params[max_region], alpha=0.1, color='red')
plt.vlines(n_params[max_test], ymin = -0.2, ymax =4, color = "black")
plt.ylim(-0.2, 4)
plt.ylabel("Train/Test loss")
plt.xlabel("Parameters")
plt.savefig("results/double_descent.pdf", format = "pdf",bbox_inches='tight')


