import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

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


def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1, label=r"$2\sigma$")
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i + 1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx', label='Training points')
    plt.legend()



def plot_gp_2D(gx, gy, mu, X_train, Y_train, title, i):
    ax = plt.gcf().add_subplot(1, 2, i, projection='3d')
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    ax.scatter(X_train[:, 0], X_train[:, 1], Y_train, c=Y_train, cmap=cm.coolwarm)
    ax.set_title(title)