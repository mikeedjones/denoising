"""
# How to use this
To use this, at the terminal:

Run privately: `export FLASK_APP=denoising/tagger.py; export FLASK_ENV=development; flask run`
Run publicly: `export FLASK_APP=denoising/tagger.py; flask run --host=0.0.0.0`

on the command line. Then go to 

`localhost:5000`

to start tagging.
"""

import matplotlib as mpl
mpl.use('Agg')

import pandas as pd
import json
import scipy as sp
from io import BytesIO
import matplotlib.pyplot as plt
from flask import Flask, Response, request, redirect
from jinja2 import Template
from . import tools, db
from pkg_resources import resource_string
app = Flask(__name__)

@app.route('/image/<category>/<number>')
def image(category, number):
    """Serves up an image. Only ever called by the browser when the tag page is loaded."""
    arr = tools.uncached_image(category, number)

    # Render the heatmap as we would normally
    ax = tools.show(arr, cbar=False)
    ax.axis('off')
    ax.figure.set_size_inches(12, 12)

    # Work out what part of the rendered plot actually contains the image data. 
    extent = mpl.transforms.Bbox([[-.5, arr.shape[0]], [arr.shape[1], -.5]])
    extent = extent.transformed(ax.transData).transformed(ax.figure.dpi_scale_trans.inverted())

    # Save out just the part of the plot with the image data to an in-memory bytestring object
    bs = BytesIO()
    ax.figure.savefig(bs, format='png', bbox_inches=extent)
    bs.seek(0)
    bs = bs.read()
    plt.clf()

    # Return the bytestring's contents as a png
    return Response(bs, mimetype='image/png')

@app.route('/tag/<category>/<number>')
def tag(category, number):
    """Serves up a tag page.
    
    The template the page is rendered from uses the category/number arguments to `render` to figure out which image to display,
    where to submit data to, etc. Look for any `{{category}}`-esque syntax in the template.
    """
    return Template(resource_string(__package__, 'tagger.j2').decode()).render(category=category, number=number)

@app.route('/tag/<category>')
def tag_category(category):
    """Redirects to the tag page for the first untagged image"""
    number = min(db.untagged(category))
    return redirect(f'/tag/{category}/{number}') 

@app.route('/')
def root():
    """Redirects to the `quadrant` tag pages"""
    return redirect('/tag/quadrant')

@app.route('/selection/<category>/<number>', methods=['POST'])
def selection(category, number):
    """Handles data submission. Only ever accessed by the JavaScript."""
    db.add(category, number, request.remote_addr, json.loads(request.data.decode()))
    return ''

