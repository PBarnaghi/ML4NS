# probability decision surface for logistic regression on a binary classification dataset
import numpy as np
import matplotlib.pyplot as plt
import torch

def pytorch_decision_boundary(
    wrapped_model, 
    X, 
    grid_resolution=200,
    ax=None,
    cmap='RdBu',
    alpha=0.5,
    levels=100,
    ):

    if ax is None:
        ax = plt.gca()

    # define bounds of the domain
    x_min, x_max = X[:, 0].min().item()-1, X[:, 0].max().item()+1
    y_min, y_max = X[:, 1].min().item()-1, X[:, 1].max().item()+1
    # define the x and y scale
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_resolution, endpoint=1),
        np.linspace(y_min, y_max, grid_resolution, endpoint=1))
    grid = np.c_[xx.ravel(), yy.ravel()]

    yhat = wrapped_model.predict(grid)
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)
   
    # plot the grid of x, y and z values as a surface
    ax.contourf(xx, yy, zz, cmap=cmap, alpha=alpha, levels=levels)
    # add a legend, called a color bar

    return ax