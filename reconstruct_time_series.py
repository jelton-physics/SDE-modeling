"""
Use Neural Network architecture to reconstruct a dynamical system from an input time series.
"""


import torch
import torch.nn as nn
from torch.nn.init import uniform_
from random import randint
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
import math

import SDE_integrate


def hankel_matrix(X, D):
    N = len(X)
    M = np.array(D*[0])
    for i in range(N-2):
        row = X[i:i+3]
        M = np.vstack((M, row))
    M = M[1:]
    return M


class shallowPLRNN(nn.Module):
    def __init__(self, M, L, K=0):
        super(shallowPLRNN, self).__init__()
        self.M = M
        self.L = L
        self.K = K
        self.autonomous = False
        if K == 0:
            self.autonomous = True
        self.init_parameters()

    def init_parameters(self):
        r1 = 1.0 / (self.L ** 0.5)
        r2 = 1.0 / (self.M ** 0.5)
        self.W1 = nn.Parameter(uniform_(torch.empty(self.M, self.L), -r1, r1))
        self.W2 = nn.Parameter(uniform_(torch.empty(self.L, self.M), -r2, r2))
        self.A = nn.Parameter(uniform_(torch.empty(self.M), a=0.5, b=0.9))
        self.h2 = nn.Parameter(uniform_(torch.empty(self.L), -r1, r1))
        self.h1 = nn.Parameter(torch.zeros(self.M))
        if self.autonomous:
            self.C = None
        else:
            r3 = 1.0 / (self.K ** 0.5)
            self.C = nn.Parameter(uniform_(torch.empty(self.M, self.K), -r3, r3))

    def forward(self, z):
        return self.A * z + torch.relu(z @ self.W2.T + self.h2) @ self.W1.T + self.h1

    def jacobian(self, z):
        """Compute the Jacobian of the model at state z. Expects z to be a 1D tensor."""
        #assert z.ndim() == 1
        return torch.diag(self.A) + self.W1 @ torch.diag(self.W2 @ z > -self.h2).float() @ self.W2

    def ext_input(self, s):
        if self.autonomous or s is None:
            return 0
        elif self.autonomous and s is not None:
            raise ValueError('Model was initialized as autonomous!')
        else:
            return s @ self.C.T

    def __call__(self, z, s=None):
        """
        Compute the next state of the model. Expects `z` and `s` to be a 2D tensor
        where the first dimension is the batch dimension.
        """
        return self.forward(z) + self.ext_input(s)

@torch.no_grad()
def generate_orbit(model, z1, T, S=None):
    """
    Generate an orbit of `model`, i.e. starting from initial condition `z1`, draw
    a trajectory of length `T` with optional external input matrix `S` of shape
    `T x K`.
    """
    if S is None:
        S = [None] * T

    z = z1
    orbit = [z]
    for t in range(T):
        z = model(z, S[t])
        orbit.append(z)
    return torch.stack(orbit)


class TimeSeriesDataset():
    """
    The dataset. It stores the observed orbit `data` of shape `T x N` and optional
    `external_inputs` of shape `T x K`. During training, this class provides batches of
    subsequences from the data.
    """
    def __init__(self, data, external_inputs=None, sequence_length=200, batch_size=16):
        self.X = torch.tensor(data, dtype=torch.float32)
        self.total_time_steps = self.X.shape[0]
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        if external_inputs is not None:
            self.S = torch.tensor(external_inputs, dtype=torch.float32)
            assert self.X.size(0) == self.S.size(0), "X and S must have the same number of time steps"
        else:
            self.S = None

    def __len__(self):
        return self.total_time_steps - self.sequence_length - 1

    def __getitem__(self, t):
        """
        Return a subsequences `x`, `y` and `s` of length `self.sequence_length`
        starting from time index `t`. `y` is the target and is simply `x` shifted
        by one time step.
        """
        x = self.X[t:t+self.sequence_length, :]
        y = self.X[t+1:t+self.sequence_length+1, :]
        if self.S is None:
            return x, y, None
        else:
            s = self.S[t:t+self.sequence_length, :]
            return x, y, s

    def sample_batch(self):
        """
        Sample a batch of sequences.
        """
        X = []
        Y = []
        S = []
        for _ in range(self.batch_size):
            idx = randint(0, len(self))
            x, y, s = self[idx]
            X.append(x)
            Y.append(y)
            S.append(s)

        if S[0] is None:
            return torch.stack(X), torch.stack(Y), None
        else:
            return torch.stack(X), torch.stack(Y), torch.stack(S)
        

def predict_sequence_using_gtf(model, x, s, alpha):
    """
    Performs an entire forward pass of the model given training sequence `x`
    and possible external inputs `s` using generalized teacher forcing with
    forcing strength `alpha`.
    """
    T = x.shape[1]
    x_ = x.permute(1, 0, 2)
    if s is not None:
        s_ = s.permute(1, 0, 2)
    else:
        s_ = [None] * T

    Z = torch.empty_like(x_)
    # initial prediction based on first data point
    z = model(x_[0], s_[0])
    Z[0] = z
    # forward pass using GTF
    for t in range(1, T):
        z_hat = x_[t]
        z = model(generalized_teacher_forcing(z, z_hat, alpha), s_[t])
        Z[t] = z
    return Z.permute(1, 0, 2)

def generalized_teacher_forcing(z, z_hat, alpha):
    return alpha * z_hat + (1 - alpha) * z

def train(model, dataset, optimizer, loss_fn, num_epochs, alpha):
    """
    Simple training method. Train `model` for `num_epochs` where one epoch is defined
    by processing a single training batch.
    """
    model.train()
    losses = []
    with trange(num_epochs, unit="batches") as pbar:
      pbar.set_description(f"Training Progress")
      for epoch in pbar:
          optimizer.zero_grad()
          x, y, s = dataset.sample_batch()
          y_hat = predict_sequence_using_gtf(model, x, s, alpha)
          loss = loss_fn(y_hat, y)
          loss.backward()
          optimizer.step()
          if epoch % 10 == 0:
              pbar.set_postfix(loss=loss.item())
          losses.append(loss.item())
    return losses

def compute_and_smooth_power_spectrum(x, smoothing):
    x_ =  (x - x.mean()) / x.std()
    fft_real = np.fft.rfft(x_)
    ps = np.abs(fft_real)**2 * 2 / len(x_)
    ps_smoothed = gaussian_filter1d(ps, smoothing)
    return ps_smoothed / ps_smoothed.sum()

def hellinger_distance(p, q):
    return np.sqrt(1-np.sum(np.sqrt(p*q)))

def power_spectrum_error(X, X_gen, smoothing):
    dists = []
    for i in range(X.shape[1]):
        ps = compute_and_smooth_power_spectrum(X[:, i], smoothing)
        ps_gen = compute_and_smooth_power_spectrum(X_gen[:, i], smoothing)
        dists.append(hellinger_distance(ps, ps_gen))
    return np.mean(dists)

# D_stsp
def calc_histogram(x, n_bins, min_, max_):
    dim_x = x.shape[1]  # number of dimensions

    coordinates = (n_bins * (x - min_) / (max_ - min_)).long()

    # discard outliers
    coord_bigger_zero = coordinates > 0
    coord_smaller_nbins = coordinates < n_bins
    inlier = coord_bigger_zero.all(1) * coord_smaller_nbins.all(1)
    coordinates = coordinates[inlier]

    size_ = tuple(n_bins for _ in range(dim_x))
    indices = torch.ones(coordinates.shape[0], device=coordinates.device)
    if 'cuda' == coordinates.device.type:
        tens = torch.cuda.sparse.FloatTensor
    else:
        tens = torch.sparse.FloatTensor
    return tens(coordinates.t(), indices, size=size_).to_dense()

def normalize_to_pdf_with_laplace_smoothing(histogram, n_bins, smoothing_alpha=10e-6):
    if histogram.sum() == 0:  # if no entries in the range
        pdf = None
    else:
        dim_x = len(histogram.shape)
        pdf = (histogram + smoothing_alpha) / (histogram.sum() + smoothing_alpha * n_bins ** dim_x)
    return pdf

def kullback_leibler_divergence(p1, p2):
    """
    Calculate the Kullback-Leibler divergence
    """
    if p1 is None or p2 is None:
        kl = torch.tensor([float('nan')])
    else:
        kl = (p1 * torch.log(p1 / p2)).sum()
    return kl

def state_space_divergence_binning(x_gen, x_true, n_bins=30):
    x_true_ = torch.tensor(x_true)
    x_gen_ = torch.tensor(x_gen)
    min_, max_ = x_true_.min(0).values, x_true_.max(0).values
    hist_gen = calc_histogram(x_gen_, n_bins=n_bins, min_=min_, max_=max_)
    hist_true = calc_histogram(x_true_, n_bins=n_bins, min_=min_, max_=max_)

    p_gen = normalize_to_pdf_with_laplace_smoothing(histogram=hist_gen, n_bins=n_bins)
    p_true = normalize_to_pdf_with_laplace_smoothing(histogram=hist_true, n_bins=n_bins)
    return kullback_leibler_divergence(p_true, p_gen).item()

def clean_from_outliers(prior, posterior):
    nonzeros = (prior != 0)
    if any(prior == 0):
        prior = prior[nonzeros]
        posterior = posterior[nonzeros]
    outlier_ratio = (1 - nonzeros.float()).mean()
    return prior, posterior, outlier_ratio

def eval_likelihood_gmm_for_diagonal_cov(z, mu, std):
    T = mu.shape[0]
    mu = mu.reshape((1, T, -1))

    vec = z - mu  # calculate difference for every time step
    vec=vec.float()
    precision = 1 / (std ** 2)
    precision = torch.diag_embed(precision).float()

    prec_vec = torch.einsum('zij,azj->azi', precision, vec)
    exponent = torch.einsum('abc,abc->ab', vec, prec_vec)
    sqrt_det_of_cov = torch.prod(std, dim=1)
    likelihood = torch.exp(-0.5 * exponent) / sqrt_det_of_cov
    return likelihood.sum(dim=1) / T

def state_space_divergence_gmm(X_gen, X_true, scaling=1.0, max_used=10000):
    time_steps = min(X_true.shape[0], max_used)
    mu_true = X_true[:time_steps, :]
    mu_gen = X_gen[:time_steps, :]

    cov_true = torch.ones(X_true.shape[-1]).repeat(time_steps, 1) * scaling
    cov_gen = torch.ones(X_gen.shape[-1]).repeat(time_steps, 1) * scaling

    mc_n = 1000
    t = torch.randint(0, mu_true.shape[0], (mc_n,))

    std_true = torch.sqrt(cov_true)
    std_gen = torch.sqrt(cov_gen)

    z_sample = (mu_true[t] + std_true[t] * torch.randn(mu_true[t].shape)).reshape((mc_n, 1, -1))

    prior = eval_likelihood_gmm_for_diagonal_cov(z_sample, mu_gen, std_gen)
    posterior = eval_likelihood_gmm_for_diagonal_cov(z_sample, mu_true, std_true)
    prior, posterior, outlier_ratio = clean_from_outliers(prior, posterior)
    kl_mc = torch.mean(torch.log(posterior + 1e-8) - torch.log(prior + 1e-8), dim=0)
    return kl_mc.item()


@torch.no_grad()
def max_lyapunov_exponent(model, T, z1, T_trans=1000, ons=1):

    # evolve for transient time Tₜᵣ
    X = generate_orbit(model, z1, T_trans)

    # initialize
    z = X[-1]
    lyap = 0
    # initialize as Identity matrix
    Q = torch.eye(model.M)

    for t in range(T):
        z=model(z)
        J = model.jacobian(z)
        Q = J @ Q

        if (t % ons == 0):

            # reorthogonalize
            Q, R = torch.linalg.qr(Q)

            # accumulate lyapunov exponents
            lyap += torch.log(torch.abs(R[0, 0])).item()

    return lyap / T


#####################################################################################################

# hyperparameters
sequence_length = 75
batch_size = 16
alpha = 0.125
learning_rate = 1e-3
hidden_size = 50
frac_test = 0.2
# dimension
N = 3

# set this to at least 5000 if you are training a model from scratch
num_epochs = 5000


# Read in time series

# S0 = np.arange(100)
X1 = 3
S0_obj = SDE_integrate.SDE_TimeSeries(X1)

# ys = S1.run_simulation_gbm()
S0 = S0_obj.run_simulation_poisson_jump()


T_series = len(S0)
T_train= math.floor((1-frac_test)*T_series)
T_test = T_series - T_train

S_train = S0[:T_train]
S_test = S0[T_train:]

# Create Hankel matrix of lags
X_train = hankel_matrix(S_train, N)
X_test = hankel_matrix(S_test, N)


# # visualize
# fig = plt.figure(figsize=(12, 10))

# ax1 = fig.add_subplot(221, projection='3d')
# ax1.plot3D(X_train[:, 0], X_train[:, 1], X_train[:, 2])
# ax1.set_title('Training trajectory')

# ax2 = fig.add_subplot(222, projection='3d')
# ax2.plot3D(X_test[:, 0], X_test[:, 1], X_test[:, 2])
# ax2.set_title('Testing trajectory')

# ax3 = fig.add_subplot(223)
# ax3.plot(X_train[1:1000, 0])
# ax3.set_xlabel("time")
# ax3.set_ylabel("x")

# ax4 = fig.add_subplot(224)
# ax4.plot(X_test[1:1000, 0])
# ax4.set_xlabel("time")
# ax4.set_ylabel("x")

# plt.tight_layout()
# plt.show()


# initialize the data set using the training data
# sequence_length and batch_size are hyperparameters, where the former has generally
# a larger impact on performance (and also runtime)
dataset = TimeSeriesDataset(X_train, sequence_length=sequence_length, batch_size=batch_size)

# initialize a shallowPLRNN without inputs (K=0), latent dimension M
# equal to the data dimension N=3, and a hidden dimension of L=50
# L is a hyperparameter which controls the expressivity of the model.
model = shallowPLRNN(M=N, L=hidden_size, K=0)

# optimizer and loss function
optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# actual training routine call, this can take some time to run!
# alpha is a hyperparameter which has a huge impact on reconstruction performance!
losses = train(model, dataset, optimizer, loss_fn, num_epochs=num_epochs, alpha=alpha)

# plot the loss
plt.plot(losses)
plt.xlabel("epochs")
plt.ylabel("MSE")
plt.title("Training loss")
plt.show()

# you can save the model here
# torch.save(model.state_dict(), "shallowPLRNN.pt")

# load pretrained model
# model = shallowPLRNN(M=3, L=50, K=0)
# model.load_state_dict(torch.load("lorenz-datasets/pretrained-shPLRNN.pt"))


# 7 Evaluation of the trained model

# lets draw an orbit of the system
x1 = torch.tensor(X_test[0, :])
orbit = generate_orbit(model, x1, T_test).numpy()

# compute the measures
D_H = power_spectrum_error(orbit, X_test, 20)
D_stsp = state_space_divergence_binning(orbit, X_test, 30)
print(f"D_H = {D_H}")
print(f"D_stsp = {D_stsp}")

# plot orbit and ground truth attractor in state space
plt.figure(figsize=(12, 7))
ax = plt.axes(projection='3d')
ax.plot3D(X_test[:, 0], X_test[:, 1], X_test[:, 2], label="True orbit")
ax.plot3D(orbit[:, 0], orbit[:, 1], orbit[:, 2], label="Generated orbit")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()


# determine the max. Lyapunov exponent. For comparison to the ground truth value, we divide by the sampling rate
# just to draw the Lorenz trajectories (\Delta t = 0.01)
max_LE = max_lyapunov_exponent(model=model, T=100000, z1=dataset.X[0, :], T_trans=10000, ons=10) / 0.01


zzz = 1
