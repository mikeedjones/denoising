import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from . import tools, db
from logging import getLogger

log = getLogger(__name__)


def bump(category, number, vals):
    ys, xs = sp.meshgrid(sp.arange(vals.shape[1]), sp.arange(vals.shape[0]))

    z0 = sp.array([
        vals.mean(), # background 
        vals.max() - vals.mean(), # height
        min(vals.shape)//2, # scale/width
        vals.shape[1]//2, # x-center
        vals.shape[0]//2]) # y-center

    def estimate(z):
        mu, h, w, x, y = z

        dist = ((xs - x)**2 + (ys - y)**2)/w**2
        bump = h*sp.exp(-dist)
        return bump, bump + mu

    def loss(z):
        return ((vals - estimate(z)[1])**2).mean()/(vals**2).mean()

    #TODO: Make this faster by passing the Jacobian
    soln = sp.optimize.minimize(loss, z0)
    if soln['success']:
        return estimate(soln['x'])[0].clip(0, None)
    else:
        log.warn(f'A bump fitting failed on {category}/{number}; setting it to zero')
        return sp.zeros_like(vals)

def bumpmap(category, number=None):
    """Returns a series mapping image number to the 'perfect image' for that source image, constructed out of Gaussians fit to
    the manual tags"""
    if number is None:
        return pd.Series({n: bumpmap(category, n) for n in db.read(category).number.unique()})

    im = tools.image(category, number)
    regions = db.regions(category, number)
    bumps = sp.zeros_like(im)
    for _, row in regions.iterrows():
        inc = bumps.__getitem__(row.extent) + bump(category, number, row.region)
        bumps.__setitem__(row.extent, inc)
    return bumps

def bumplist(category, number=None):
    if number is None:
        return pd.concat({n: bumplist(category, n) for n in db.read(category ).number.unique()})

    regions = db.regions(category, number)
    bumps = []
    for _, row in regions.iterrows():
        bumps.append(bump(category, number, row.region))
    return pd.Series(bumps)

def example(category, number):
    im = tools.image('quadrant', number)
    bumps = bumpmap('quadrant', number)
    tools.show({'im': im, 'bumps': bumps, 'resid': im - bumps})