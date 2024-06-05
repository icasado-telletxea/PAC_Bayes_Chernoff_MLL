
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import latex_format

# Activate Latex format for matplotlib
latex_format()



labels = np.loadtxt("models/model_labels.txt", delimiter=" ", dtype = str)
n_params = np.loadtxt("models/n_params.txt")

subset = "last_layer"
hessian = "kron"

# Read the csv
results = pd.read_csv(f"results/2laplace_{subset}_{hessian}_results.csv")
Gibbs_losses_train = results.loc[:, "gibbs loss train"].to_numpy()
Bayes_losses = results.loc[:, "bayes loss"].to_numpy()
Bound = results.loc[:, "gibbs loss train"].to_numpy() + results.loc[:, "inverse rate"].to_numpy()

# Window size for smoothing
N = 7
train_loss_ = *([Gibbs_losses_train[0]]*(N//2)), *Gibbs_losses_train, *([Gibbs_losses_train[-1]]*(N//2))
test_loss_ = *([Bayes_losses[0]]*(N//2)), *Bayes_losses, *([Bayes_losses[-1]]*(N//2))
Bound_ = *([Bound[0]]*(N//2)), *Bound, *([Bound[-1]]*(N//2))

train_loss_ = np.convolve(train_loss_, np.ones(N)/N, mode = "valid")
test_loss_ = np.convolve(test_loss_, np.ones(N)/N, mode = "valid")
Bound_ = np.convolve(Bound_, np.ones(N)/N, mode = "valid")

plt.rcParams['figure.figsize'] = (16, 8)

jet = plt.colormaps["Set2"]
lw = 5
plt.plot(n_params, Gibbs_losses_train, linewidth = lw, color = jet(1), alpha = 0.2)
plt.plot(n_params, Bayes_losses, linewidth = lw, color = jet(2), alpha = 0.2)
plt.plot(n_params, Bound, linewidth = lw, color = jet(3), alpha = 0.2)

plt.plot(n_params, train_loss_,  linewidth =lw, color = jet(1))
plt.plot(n_params, test_loss_, linewidth = lw, color = jet(2))
plt.plot(n_params, Bound_, linewidth = lw, color = jet(3))

plt.scatter(n_params, Gibbs_losses_train, marker = "*", label = "Train loss", color = jet(1), s = 100)
plt.scatter(n_params, Bayes_losses, marker = "o", label = "Test loss", color = jet(2), s = 100)
plt.scatter(n_params, Bound, marker = "o", label = "Bound", color = jet(3), s = 100)

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
plt.ylim(-0.2, 3.9)
plt.ylabel("Train/Test loss")
plt.xlabel("Parameters")
plt.savefig(f"results/double_descent_{subset}_{hessian}.pdf", format = "pdf",bbox_inches='tight')