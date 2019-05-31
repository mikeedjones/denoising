import pandas as pd
import scipy as sp
from . import nn, gaussians, db
import seaborn as sns
import skimage
import skimage.feature
import seaborn as sns

def counts():
    bumpmaps = nn.bumpmaps('quadrant')
    bumplist = gaussians.bumplist('quadrant')
    average_energy = bumplist.apply(sp.sum).loc[lambda s: s != 0].mean()
    total_energy = bumpmaps.sum(1).sum(1)
    counts = total_energy / average_energy
    sns.distplot(counts)

def peaklist(category, number=None):
    """Uses peak finding to try and count the number of spots in each image"""
    if number is None:
        return pd.concat({n: peaklist(category, n) for n in db.read(category).number.unique()})

    bumps = nn.bumpmaps(category, number)
    blurred = sp.ndimage.gaussian_filter(bumps, 3) # This'll depress the maxes somewhat
    maxes = skimage.feature.peak_local_max(blurred, min_distance=2)

    peaks = blurred[maxes[:, 0], maxes[:, 1]]
    maxes = maxes[peaks > .0025]
    peaks = peaks[peaks > .0025]

    return pd.DataFrame({'peak': peaks, 'row': maxes[:, 0], 'col': maxes[:, 1]})

def example():
    ps = peaklist('quadrant')
    sns.distplot(ps.peak)