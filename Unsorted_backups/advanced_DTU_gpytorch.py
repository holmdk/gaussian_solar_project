#!/usr/bin/env python
# coding: utf-8

# # Gaussian Process Regression using GPyTorch

# In[1]:


import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from sklearn.preprocessing import StandardScaler

import math
import torch
import gpytorch
from matplotlib import pyplot as plt

test = True


def nc_to_pd(data, name=None, save=True, save_folder=None):
    data = data.to_dataframe()
    data.index = pd.to_datetime(data.index, utc=True)
    data = data[['SIS']]

    data.dropna(inplace=True)

    if save == True:
        data.to_csv(save_folder + name + '.csv')

    return data


def prepare_simulated(data, time_idx):
    # we define timestamps as our index in utc time
    data['time'] = pd.to_datetime(data.time, utc=True)
    data.set_index('time', inplace=True)

    # we remove missing observations
    data = data[data['missing'] == 0]

    # And then we only select time and electricty as the columns
    data = data[['electricity']]

    # and finally, we select the timestamps defined for our satellite dataset
    data = data.loc[time_idx]

    return data
# In[2]:


if sys.platform == 'win32':
    save_folder = "D:/Google Drive/Dropbox_Backup/Post Study/Industrial/Courses/DTU Advanced Machine Learning, 2020/gaussian_solar_project/data/"
else:
    save_folder = "/home/local/DAC/ahn/Documents/dcwis.solar/dcwis/solar/reports/dtu/"


# In[3]:


meuro_sim = pd.read_csv(save_folder + 'meuro_simulated.csv', skiprows=[0,1,2])
neuhardenberg_sim = pd.read_csv(save_folder + 'neuhardenberg_simulated.csv', skiprows=[0,1,2])
templin_sim = pd.read_csv(save_folder + 'templin_simulated.csv', skiprows=[0,1,2])


# In[4]:


meuro_irradiance = pd.read_csv(save_folder + 'irradiance_meuro_data.csv').set_index('time')
meuro_irradiance.index = pd.to_datetime(meuro_irradiance.index, utc=True)

neuhardenberg_irradiance = pd.read_csv(save_folder + 'irradiance_neuhardenberg_data.csv').set_index('time')
neuhardenberg_irradiance.index = pd.to_datetime(neuhardenberg_irradiance.index, utc=True)

templin_irradiance = pd.read_csv(save_folder + 'irradiance_templin_data.csv').set_index('time')
templin_irradiance.index = pd.to_datetime(templin_irradiance.index, utc=True)


# In[5]:


# prepare our simulated dataframes
meuro_sim = prepare_simulated(meuro_sim, meuro_irradiance.index)
neuhardenberg_sim = prepare_simulated(neuhardenberg_sim, neuhardenberg_irradiance.index)
templin_sim = prepare_simulated(templin_sim, templin_irradiance.index)


# In[6]:


# Let us start considering the Meuro dataset

if test:
    # Training data is 100 points in [0,1] inclusive regularly spaced
    x = torch.linspace(0, 1, 100)
    # True function is sin(2*pi*x) with Gaussian noise
    y = torch.sin(x * (2 * math.pi)) + torch.randn(x.size()) * math.sqrt(0.04)
else:
    # In[7]:
    x = meuro_irradiance.values.reshape(-1, 1)
    y = meuro_sim.values.reshape(-1, 1)

    # In[8]:

    # We start by standardizing the data, note we save the standardizers in case we want to add back the mean
    x_scaler = StandardScaler().fit(x)
    y_scaler = StandardScaler().fit(y)

    x = x_scaler.transform(x)
    y = y_scaler.transform(y)

    x = torch.from_numpy(x.ravel())
    y = torch.from_numpy(y.ravel())


# In[9]:


# and plot it
# plt.figure(figsize=(12,10))
# plt.plot(x, y, 'rx', color='black')
# plt.ylabel('Normalized level of solar production')
# plt.xlabel('Normalized level of solar irradiance')
# plt.show()



# In[11]:


# We convert variables to torch tensors

# 
# # Training data is 100 points in [0,1] inclusive regularly spaced
# x = torch.linspace(0, 1, 100)
# # True function is sin(2*pi*x) with Gaussian noise
# y = torch.sin(x * (2 * math.pi)) + torch.randn(x.size()) * math.sqrt(0.04)
# # Xtest = torch.tensor(Xtest)


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(x, y, likelihood)



# In[13]:
# Find optimal model hyperparameters
model.train()
likelihood.train()


# In[14]:
# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# In[15]:
# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


# In[16]:
training_iter = 50

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(x)
    # Calc loss and backprop gradients
    loss = -mll(output, y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()


# In[ ]:





# In[ ]:


# Make predictions


# In[ ]:

# We start by standardizing the data, note we save the standardizers in case we want to add back the mean
if not test:
    x_test = neuhardenberg_irradiance.values.reshape(-1, 1)
    y_test = neuhardenberg_sim.values.reshape(-1, 1)

    x_test_scaler = StandardScaler().fit(x_test)
    y_test_scaler = StandardScaler().fit(y_test)

    x_test = x_test_scaler.transform(x_test)
    y_test = y_test_scaler.transform(y_test)


    # We convert variables to torch tensors
    x_test = torch.from_numpy(x_test.ravel())
    y_test = torch.from_numpy(y_test.ravel())


# f_preds = model(x_test)
# y_preds = likelihood(model(x_test))

# f_mean = f_preds.mean
# f_var = f_preds.variance
# f_covar = f_preds.covariance_matrix
# f_samples = f_preds.sample(sample_shape=torch.Size(1000,))


# In[ ]:


# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    if test:
        x_test = torch.linspace(0, 1, 51)
    observed_pred = likelihood(model(x_test))


# In[ ]:


with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(x.numpy(), y.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(x_test.numpy(), observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(x_test.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# # We fit the GP
# mu, covariance = fit_predictive_GP(x, y, Xtest, lengthscale, kernel_var, noise_var)
# std = np.sqrt(np.diag(covariance))


# log_like_initial = loglike(x, y, squared_exponential_kernel, lengthscale, kernel_var, noise_var)


# In[ ]:


# # We plot the initial GP
# plt.figure(figsize=(12,10))
# plt.plot(x, y, 'rx', label='Training points')
# plt.gca().fill_between(Xtest.flat, mu.reshape(-1) - 2 * std, mu.reshape(-1) + 2 * std,  color='lightblue', alpha=0.4, label=r"$2\sigma$")
# plt.plot(Xtest, mu, 'blue', label=r"$\mu$")
# plt.legend()
# plt.show() 

# print('The log likelihood for this fit is %.4f' % -log_like_initial)


# In[ ]:





# In[ ]:





# In[ ]:




