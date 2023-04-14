import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import requests
from io import BytesIO

cmap = [(0, '#ddf'), (0.5, '#0af'), (1, '#80a')]
cmap = cm.colors.LinearSegmentedColormap.from_list('Custom', cmap, N=256)


def plotProg(hc, hm, opt):
    fig = plt.figure()
    plt.plot(hc, color='b', linestyle='-', label='best')
    plt.plot(hm, color='m', linestyle='--', label='mean')
    plt.axhline(opt, color='g', linestyle='-', label='optimal')
    plt.xlabel('Number of iteration')
    plt.ylabel('Value')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.show()

def plotSearchSpace(func, hs, sol, best, cmap=cmap):
    X_domain = [-100, 100]
    Y_domain = [-100, 100]
    X, Y = np.linspace(*X_domain, 100), np.linspace(*Y_domain, 100)
    X, Y = np.meshgrid(X, Y)
    XY = np.array([X, Y])
    Z = np.apply_along_axis(func.evaluate, 0, XY)
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.contourf(X, Y, Z, levels=30, cmap=cmap)
    ax1.plot(hs[:, 0], hs[:, 1], 'k.', label='pop')
    ax1.plot(sol[0], sol[1], 'ro', label='optimal')
    ax1.plot(best[0], best[1], 'gx', label='best')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.set_aspect(aspect='equal')

    X, Y = np.linspace(*X_domain, 100), np.linspace(*Y_domain, 100)
    X, Y = np.meshgrid(X, Y)
    XY = np.array([X, Y])
    Z = np.apply_along_axis(func.evaluate, 0, XY)
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(1, 1, 1, projection='3d')
    ax2.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, rstride=1, cstride=1, antialiased=False)
    ax2.contour(X, Y, Z, zdir='z', levels=30, offset=np.min(Z), cmap=cmap)
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('f(x,y)')
    ax2.xaxis.set_tick_params(labelsize=8)
    ax2.yaxis.set_tick_params(labelsize=8)
    ax2.zaxis.set_tick_params(labelsize=8)
    plt.show()
    