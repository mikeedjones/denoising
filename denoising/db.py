"""Stores the manual tags generated from the tagger"""

from tqdm import tqdm
import matplotlib as mpl
import sqlite3
import pandas as pd
import scipy as sp
from contextlib import contextmanager
from pathlib import Path
from . import tools

VERSION = '0.1'

@contextmanager
def connection():
    Path('data').mkdir(exist_ok=True, parents=True)
    with sqlite3.connect('data/tags.sqlite') as conn:
        conn.cursor().execute('''create table if not exists Tags (
                            id integer primary key,
                            category text,
                            number integer,
                            x1 real,
                            y1 real,
                            x2 real,
                            y2 real,
                            width real,
                            height real,
                            date text,
                            ip text,
                            version text)''')
        yield conn

def add(category, number, ip, data):
    rows = pd.DataFrame(data)
    rows['category'] = category
    rows['number'] = number
    rows['ip'] = ip
    rows['date'] = pd.Timestamp.now()
    rows['version'] = VERSION
    with connection() as conn:
        rows.to_sql('Tags', conn, if_exists='append', index=False)

def read(category=None, number=None):
    if category is not None:
        return read(None, number).query(f'category == "{category}"')
    if number is not None:
        return read(category, None).query(f'number == {number}')

    with connection() as conn:
        return pd.read_sql('select * from Tags', conn, index_col='id', parse_dates=['date'])

def untagged(category):
    return set(range(1, tools.COUNTS[category])) - set(read().number)

def show(category, number):
    arr = tools.image(category, number)
    ax = tools.show(arr)

    rows = read(category, number)

    boxes = {}
    boxes['w'] = (rows.x2 - rows.x1)/rows.width*arr.shape[1]
    boxes['h'] = (rows.y2 - rows.y1)/rows.height*arr.shape[0]
    boxes['y'] = rows.y1/rows.height*arr.shape[0]
    boxes['x'] = rows.x1/rows.width*arr.shape[1]
    boxes = pd.concat(boxes, 1)

    rects = list(boxes.apply(lambda r: mpl.patches.Rectangle((r.x, r.y), r.w, r.h), axis=1))

    ax.add_collection(mpl.collections.PatchCollection(rects, alpha=.25, color='k'))

    return ax

def extent(arr, row):
    t = int(sp.floor(row.y1/row.height*arr.shape[0]))
    l = int(sp.floor(row.x1/row.width*arr.shape[1]))
    b = int(sp.ceil(row.y2/row.height*arr.shape[0]))
    r = int(sp.ceil(row.x2/row.width*arr.shape[1]))
    return (slice(t, b), slice(l, r))

def regions(category, number=None):
    if number is None:
        return pd.concat([regions(category, n) for n in read().number.unique()])
    im = tools.image(category, number)
    results = []
    for _, row in read(category, number).iterrows():
        ex = extent(im, row)
        reg = im.__getitem__(ex).copy()
        results.append((row.number, ex, reg))
    return pd.DataFrame(results, columns=['number', 'extent', 'region'])

def mask(category, number=None):
    if number is None:
        return pd.Series({n: mask(category, n) for n in read().number.unique()})
    im = tools.image(category, number)
    m = sp.zeros_like(im, dtype=bool)
    for _, row in read(category, number).iterrows():
        m.__setitem__(extent(im, row), True)
    return m