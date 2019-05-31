import scipy as sp
import scipy.ndimage
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

COUNTS = {'full': 800, 'quadrant': 250}

def show(arr, ax=None, cbar=True, **kwargs):
    if isinstance(arr, dict):
        lim = max(sp.fabs(a).max() for a in arr.values())
        fig, axes = plt.subplots(1, len(arr))
        for (k, v), ax in zip(arr.items(), axes):
            show(v, ax, vmin=-lim, vmax=lim, cbar=False)
            ax.set_title(k)
        return axes             
    
    ax = plt.subplot() if ax is None else ax
    lim = sp.fabs(arr).max()
    kwargs = {'vmin': -lim, 'vmax': lim, 'cmap': 'RdBu', **kwargs}
    im = ax.imshow(arr, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    if cbar:
        plt.colorbar(im, fraction=.02)
    
    return ax

def show3d(arr):
    #TODO: x-axis is still back-to-front; need to change origin
    X = sp.arange(0, arr.shape[1])[::-1]
    Y = sp.arange(0, arr.shape[0])
    X, Y = sp.meshgrid(X, Y)
    Z = arr

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='Blues',
                        linewidth=0, antialiased=True)
                        
    ax.set_xlabel('x')
    ax.set_ylabel('y')

_quadrant_background = None
def quadrant_background():
    global _quadrant_background
    if _quadrant_background is None:
        return (sp.genfromtxt('data/bg/bg.asc', delimiter='\t')[:, 1:-1]/200 - 398)/400
    return _quadrant_background

def _quadrant_image(i):
    return (sp.genfromtxt(f'data/900V/900V_5kV{i:04d}.asc', delimiter='\t')[:, 1:-1] - 398)/400

_quadrant = None
def quadrant_image(i=None):
    global _quadrant
    if _quadrant is None:  
        path = Path('cache/quadrant.npy')
        if not path.exists():
            ims = [_quadrant_image(i) for i in range(1, COUNTS['quadrant'] + 1)]
            sp.save(path, sp.stack(ims))
        _quadrant = sp.load(path)
    ims = _quadrant[i-1] if i is not None else _quadrant
    return ims

def _full_image(i):
    return (sp.genfromtxt(f'data/483_53/ac_483_53_{i:04d}.asc', delimiter='\t')[:, 1:-1] - 398)/400
    
_full_background = None
def full_background():
    global _full_background
    if _full_background is None:
        return (sp.genfromtxt('data/bg-2/bg.asc', delimiter='\t')[:, 1:-1]/200 - 398)/400
    return _full_background

_full = None
def full_image(i=None):
    global _full
    if _full is None:  
        path = Path('cache/full.npy')
        if not path.exists():
            ims = [_full_image(i) for i in range(1, COUNTS['full'] + 1)]
            sp.save(path, sp.stack(ims))
        _full = sp.load(path)
    ims = _full[i-1] if i is not None else _full
    return ims

def uncached_image(category, number):
    return globals()[f'_{category}_image'](int(number))

def image(category, number):
    return globals()[f'{category}_image'](int(number))
