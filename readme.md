## Install
  * `wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh`
  * `chmod u+x https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh`
  * `conda install scipy numpy cvxpy matplotlib ipython flask tqdm scikit-image --yes`
  * `conda install cvxpy -c conda-forge --yes`
  * `conda install pytorch -c pytorch`

## Usage
If you've got a set of cache files already, you should be able to start a Python console, `from denoising import *`, and then run `example()`.

Otherwise, to run the whole pipeline end-to-end you need to 
  * Run the tagger and tag a bunch of images. There's instructions on how to run the tagger at the top of its file. 
  * Call `nn.train("quadrant")` to train a neural network based on your tags. This'll take half an hour or so.
  * Call `peaklist("quadrant")` to generate the peaklist. This'll take an hour the first time as all the perfect images are generated and cached. 

## Explanation
The idea is
  * Tag a thousand or so spots manually using [the tagger](denoising/tagger.py).
    * The tagger is a [Flask](http://flask.pocoo.org/) server which uses the [jinja2](http://jinja.pocoo.org/docs/2.10/) templater to convert the [tagger.j2](denoising/tagger.j2) HTML template into the webpages you're presented with.
    * The template makes use of a [third party script](http://odyniec.net/projects/imgareaselect/) to provide area selection, and then adds a small amount of my own script (the stuff in the `script` tag) to keep track of the selected areas and watch out for enter/backspace keypresses. 
    * When you hit enter, the script in the template gathers up the selections and sends them back to the server. The server saves them in a [sqlite3](https://docs.python.org/3/library/sqlite3.html) database.
  * Fit a Gaussian [to each tagged spot](denoising/gaussians.py) and combine the fits to create a perfect, noise-free version of each image.
  * [Train a neural network](denoising/nn.py) to predict each pixel of the perfect image from the corresponding pixels nearby in the original image.
    * The best textbook for this stuff is [Deep Learning](https://www.deeplearningbook.org/), particularly the bits on convolutional neural networks. For something lighter-weight and more directly useful, read the [PyTorch tutorials](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) or just Google for 'MNIST tutorial', which will likely use a very similar methodology.
    * The neural network is implemented in [PyTorch](https://pytorch.org/), and is loosely based off the [MNIST example](https://github.com/pytorch/examples/tree/master/mnist). The MNIST problem is an ancient [single-digit handwriting recognition task](https://en.wikipedia.org/wiki/MNIST_database).
    * The NN takes a 17-pixel (2*8 + 1) 'receptive field' around each pixel in the source image and tries to predict the value of the central pixel in the perfect image. Most of the code in [nn.py](denoising/nn.py) is to turn the input bumpmaps and images into (receptive field, perfect pixel value) pairs that the NN can ingest.
    * The NN is trained by a form of [quasi-Newton](https://en.wikipedia.org/wiki/Quasi-Newton_method) stochastic gradient descent called [ADAM](https://arxiv.org/abs/1711.05101).
  * [Count peaks](denoising/__init__.py) in the perfect images the neural network generates. 
