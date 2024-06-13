
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
p = 100.0

# Read the csv
results = pd.read_csv(f"results/laplace_{subset}_{hessian}_{p}_results.csv")
Gibbs_losses_train = results.loc[:, "gibbs loss train"].to_numpy()
Gibbs_losses = results.loc[:, "gibbs loss"].to_numpy()
Bayes_losses = results.loc[:, "bayes loss"].to_numpy()
log_marginal = results.loc[:, "neg log marginal laplace"].to_numpy()
elbo = results.loc[:, "neg log marginal"].to_numpy()
Bound = results.loc[:, "gibbs loss train"].to_numpy() + results.loc[:, "inverse rate"].to_numpy()

# fig, axis = plt.subplots(1, 3)
# axis[0].scatter(Bayes_losses, Gibbs_losses, label = "Gibbs vs Bayes")
# axis[1].scatter(Bayes_losses, log_marginal, label = "Log Marginal vs Bayes")
# axis[2].scatter(Bayes_losses, Bound, label = "Bound vs Bayes")
# plt.legend()
# plt.show()
# plt.clf()

#Bound = results.loc[:, "gibbs loss train"].to_numpy() + np.sqrt(0.5 * results.loc[:, "normalized KL"].to_numpy() * results.loc[:, "variance"].to_numpy())

# Window size for smoothing
N = 7
train_loss_ = *([Gibbs_losses_train[0]]*(N//2)), *Gibbs_losses_train, *([Gibbs_losses_train[-1]]*(N//2))
bayes_loss_ = *([Bayes_losses[0]]*(N//2)), *Bayes_losses, *([Bayes_losses[-1]]*(N//2))
gibbs_loss_ = *([Gibbs_losses[0]]*(N//2)), *Gibbs_losses, *([Gibbs_losses[-1]]*(N//2))

Bound_ = *([Bound[0]]*(N//2)), *Bound, *([Bound[-1]]*(N//2))

train_loss_ = np.convolve(train_loss_, np.ones(N)/N, mode = "valid")
bayes_loss_ = np.convolve(bayes_loss_, np.ones(N)/N, mode = "valid")
gibbs_loss_ = np.convolve(gibbs_loss_, np.ones(N)/N, mode = "valid")

Bound_ = np.convolve(Bound_, np.ones(N)/N, mode = "valid")

plt.rcParams['figure.figsize'] = (16, 8)

jet = plt.colormaps["Set2"]
lw = 5
plt.plot(n_params, Gibbs_losses_train, linewidth = lw, color = jet(1), alpha = 0.2)
plt.plot(n_params, Bayes_losses, linewidth = lw, color = jet(2), alpha = 0.2)
plt.plot(n_params, Gibbs_losses, linewidth = lw, color = jet(3), alpha = 0.2)
plt.plot(n_params, Bound, linewidth = lw, color = jet(4), alpha = 0.2)
plt.plot(n_params, log_marginal, linewidth = lw, color = jet(5), alpha = 0.2)
plt.plot(n_params, elbo, linewidth = lw, color = jet(6), alpha = 0.2)


plt.plot(n_params, train_loss_,  linewidth =lw, color = jet(1))
plt.plot(n_params, bayes_loss_, linewidth = lw, color = jet(2))
plt.plot(n_params, gibbs_loss_, linewidth = lw, color = jet(3))
plt.plot(n_params, Bound_, linewidth = lw, color = jet(4))
plt.plot(n_params, log_marginal, linewidth = lw, color = jet(5))
plt.plot(n_params, elbo, linewidth = lw, color = jet(6))

plt.scatter(n_params, Gibbs_losses_train, marker = "o", label = "Train loss", color = jet(1), s = 100)
plt.scatter(n_params, Bayes_losses, marker = "o", label = "Bayes loss", color = jet(2), s = 100)
plt.scatter(n_params, Gibbs_losses, marker = "o", label = "Gibbs loss", color = jet(3), s = 100)
plt.scatter(n_params, Bound, marker = "o", label = "Bound", color = jet(4), s = 100)
plt.scatter(n_params, log_marginal, marker = "o", label = "Neg Log Marginal Laplace", color = jet(5), s = 100)
plt.scatter(n_params, elbo, marker = "o", label = "Neg Log Marginal Upper Bound", color = jet(6), s = 100)

max_test = np.argmax(bayes_loss_)
mask = np.isclose(bayes_loss_, bayes_loss_[max_test], 0.05)
min_region = np.where(mask == True)[0][0]
max_region = np.where(mask == True)[0][-1]

plt.annotate("Classical Regime", xy = (0, 3.5))
plt.annotate("Variance-Bias Tradeoff", xy = (10000, 3.2), fontsize = 20)

plt.annotate("Modern Regime",xy = (n_params[max_region]+200000, 3.5), xytext = (n_params[max_test]+300000, 3.5))
plt.annotate("Larger is Better", xy = (n_params[max_test]+350000, 3.2), fontsize = 20)

plt.annotate("Interpolation Threshhold",xy = (n_params[max_test], 0.6), xytext = (n_params[max_test]+100000, 0.5), arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate("Critical Region",xy = (n_params[max_test]+100000, 1.1), xytext = (n_params[max_test]+200000, 1), arrowprops=dict(color='tomato', shrink=0.05))


# plt.annotate(r"\textbf{Gibbs Train loss}",xy = (n_params[max_test]+500000, 0.2), xytext = (n_params[max_test]+700000, 0.2), color = jet(1))
# plt.annotate(r"\textbf{Bayes loss}",xy = (n_params[max_test]+500000, 1.1), xytext = (n_params[max_test]+700000, 1.1), color = jet(2))
# plt.annotate(r"\textbf{Gibbs loss}",xy = (n_params[max_test]+500000, 1.1), xytext = (n_params[max_test]+700000, 2.4), color = jet(3))
# plt.annotate(r"\textbf{Bound}",xy = (n_params[max_test]+500000, 1.1), xytext = (n_params[max_test]+500000, 2.5), color = jet(4))
# plt.annotate(r"\textbf{Neg Log Marginal Lap.}",xy = (n_params[max_test]+500000, 0.5), xytext = (n_params[max_test]+500000, 0.8), color = jet(5))
# plt.annotate(r"\textbf{Neg Log Marginal Upper Bound}",xy = (n_params[max_test]+500000, 1.1), xytext = (n_params[max_test]+400000, 2.9), color = jet(6))

plt.axvspan(n_params[min_region], n_params[max_region], alpha=0.1, color='red')
plt.vlines(n_params[max_test], ymin = -0.2, ymax =4, color = "black")
plt.ylim(-0.2, 3.9)
plt.hlines(0., xmin = n_params[0], xmax = n_params[-1], color = "black")
plt.legend(loc = "lower center", ncol=3, fancybox=True, shadow=True,  bbox_to_anchor=(0.47, -0.4))
plt.ylabel("Train/Test loss")
plt.xlabel("Parameters")
plt.title(rf"Double Descent LLA KFAC $p={{{p}}}$")
plt.savefig(f"results/double_descent_{subset}_{hessian}_{p}.pdf", format = "pdf",bbox_inches='tight')