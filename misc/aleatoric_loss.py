import torch
from torch import nn
import misc.config as config


class AleatoricCrossEntropyLoss(nn.Module):
   def __init__(self):
      super(AleatoricCrossEntropyLoss, self).__init__()
      self.elu = nn.ELU()
      self.categorical_crossentropy = undistorted_cross_entropy()
      self.T = config.monte_carlo_simulations
      self.num_classes = config.ans_classes

   def forward(self, logit_var, pred, true):
      std = torch.sqrt(logit_var)
      variance = logit_var

      variance_diff = torch.exp(variance) - torch.ones_like(variance)
      variance_depressor = torch.mean(variance_diff)
      undistorted_loss = self.categorical_crossentropy(pred, true)

      dist = torch.distributions.Normal(torch.zeros_like(std), std)

      monte_carlo_results = 0
      monte_carlo_results_gce = 0
      for i in range(0, self.T):
         mc_gce_loss, mc_gce_diff_loss = self.gaussian_categorical_crossentropy(pred, true, dist, undistorted_loss)
         monte_carlo_results = monte_carlo_results + mc_gce_diff_loss
         monte_carlo_results_gce = monte_carlo_results_gce + mc_gce_loss
      variance_loss = (monte_carlo_results / self.T)
      gce_loss = (monte_carlo_results_gce / self.T)

      return gce_loss, variance_loss, undistorted_loss, variance_depressor

   def gaussian_categorical_crossentropy(self, pred, true, dist, undistorted_loss):
      std_samples = (dist.sample_n(self.num_classes)).transpose(0, 1)
      std_samples = std_samples.view(std_samples.size(0), -1)

      distorted_loss = self.categorical_crossentropy(pred + std_samples, true)
      diff = undistorted_loss - distorted_loss

      return distorted_loss, -self.elu(diff)

   def categorical_cross_entropy(true, pred):
      return np.sum(true * np.log(pred), axis=1)


class undistorted_cross_entropy(nn.Module):
   def __init__(self):
      super(undistorted_cross_entropy, self).__init__()
      self.log_softmax = nn.LogSoftmax().cuda()

   def forward(self, pred, true):
      nll = -self.log_softmax(pred)
      return (nll * true / 10).sum(dim=1).mean()
