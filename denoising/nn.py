import pandas as pd
import scipy as sp
import torch 
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import pickle
from . import db, tools, gaussians
import matplotlib.pyplot as plt

BUMP_SCALE = .02 # Estimated from nonzero bumps

def mirror(im, radius):
    return sp.pad(im, ((radius, radius), (radius, radius)), mode='reflect')

class Dataset(torch.utils.data.Dataset):

    def __init__(self, category, bumps=None, numbers=None, radius=8):
        """This is a PyTorch object that helps generate 'minibatches' of examples to train/run a neural network on. 
        It's documented [here](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html).
        """
        self.radius = radius

        if bumps is None:
            self.numbers = numbers
        else:
            self.bumps = bumps
            self.numbers = bumps.index

        # Pad out the edges of each image with a `radius` reflection so that we can tag the pixels there.
        self.images = pd.Series([mirror(tools.image(category, n), radius) for n in self.numbers], self.numbers)

        example = tools.image(category, self.images.index[0])
        self.width = example.shape[1]
        self.height = example.shape[0]
        self.centers_per_image = self.width*self.height

    def __len__(self):
        return len(self.images)*self.centers_per_image

    def __getitem__(self, i):
        """Implements the [`[i]` syntax for this object, returning the i-th example. In our case that's the i-th pixel from a perfect image
        and the region around that pixel in the source image."""
        image_index = self.numbers[i//self.centers_per_image]
        remainder = i - self.centers_per_image*(i//self.centers_per_image)

        row = remainder//self.width
        col = (remainder - self.width*(remainder//self.width))

        image = self.images.loc[image_index]
        region = image[row:row+2*self.radius+1, col:col+2*self.radius+1]

        # Neural networks work best when the inputs are mean-zero, std-one. It doesn't have to be exact, so here I've just estimated
        # some values from a small sample.
        mu, std = 0.0255, 0.006 
        region = ((region - mu)/std)

        batch = {'region': region[None, :, :].astype(sp.float32), 'row': row, 'col': col}
        if hasattr(self, 'bumps'):
            batch['label'] = sp.float32(self.bumps.loc[image_index][row, col]/BUMP_SCALE)

        return batch

class Net(nn.Module):
    def __init__(self):
        """Largely derived from the [PyTorch MNIST example](https://github.com/pytorch/examples/blob/master/mnist/main.py)
        """
        self.name = 'bumps' # the name the network params will be stored under
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(80, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = func.relu(func.max_pool2d(self.conv1(x), 2))
        x = func.relu(self.conv2(x))
        x = x.view(-1, 80)
        x = func.relu(self.fc1(x))

        # This is non-standard. There aren't many neural nets that predict values on [0, inf), so I picked a smooth 
        # function which has that range. 
        x = torch.exp(self.fc2(x))
        return x[:, 0]

def save(losses, model):
    i = len(losses)
    folder = Path(f'cache/nn/{model.name}/{i}')
    folder.mkdir(exist_ok=True, parents=True)

    torch.save(model.state_dict(), folder / 'model.torch')
    (folder / 'losses.pkl').write_bytes(pickle.dumps(losses))

def load(name, epoch=None):
    if epoch is None:
        epoch = max([int(str(p.name)) for p in Path(f'cache/nn/{name}').iterdir()])

    folder = Path(f'cache/nn/{name}/{epoch}')
    model = Net()
    model.load_state_dict(torch.load(folder / 'model.torch'))

    losses = pickle.loads((folder / 'losses.pkl').read_bytes())
    return losses, model

def test(test_loader, model):
    model.eval()
    loss = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='test'):
            loss.append(func.mse_loss(model(batch['region']), batch['label']).detach().numpy())
    return sp.array(loss).mean()

def train(category):
    """There's a lot going on in this function. It's standard boilerplate for the most part, and you should be able
    to work through it by following along with the [PyTorch tutorials](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
    """
    # Load our perfect images
    bumps = gaussians.bumpmap(category)

    # Split off the last five examples as a test (~ validation) set
    train_bumps = bumps.iloc[:-5]
    test_bumps = bumps.iloc[-5:]
    
    # Set up data loaders for the train and test sets.
    train_loader = torch.utils.data.DataLoader(Dataset('quadrant', train_bumps), batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(Dataset('quadrant', test_bumps), batch_size=64, shuffle=False, num_workers=2)

    # Set up the net and the optimizer
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=.001)

    train_losses, test_losses = [], []
    # Loop over the training data, checking things on our test set at the end of each loop (aka epoch).
    for epoch in range(5): # 5 epochs seems enough empirically
        model.train()
        for batch in tqdm(train_loader, desc='train'):
            optimizer.zero_grad()
            yhat = model(batch['region'])
            loss = func.mse_loss(yhat, batch['label'])
            loss.backward()
            optimizer.step()
            
            train_losses.append(float(loss))

        test_losses.append(test(test_loader, model))
        print(f'Loss on epoch #{epoch} is {test_losses[-1]:.3f}')

        # Save the model down.
        save(test_losses, model)

def evaluate(category, number, model):
    """Uses a NN to generate a perfect image from a previously-unseen source image"""
    loader = torch.utils.data.DataLoader(Dataset(category, numbers=[number]), batch_size=64, shuffle=False, num_workers=2)
    bs = []
    rows, cols = [], []
    for batch in tqdm(loader, desc='eval'):
        rows.extend(batch['row'].detach().numpy())
        cols.extend(batch['col'].detach().numpy())
        bs.extend(BUMP_SCALE*model(batch['region']).detach().numpy())
    bs = sp.array(bs)
    rows, cols = sp.array(rows), sp.array(cols)
    
    im = tools.image(category, number)
    bumps = sp.full_like(im, sp.nan)
    bumps[rows, cols] = bs

    return im, bumps

# The model and epoch to use. Epoch #3 had the best test loss in this case.
MODEL = ('bumps', 3)

def bumpmaps(category, number=None):
    """Uses a NN to generate a perfect image, and caches the result so it'll be fast to load next time"""

    if number is None:
        return sp.stack([bumpmaps(category, n) for n in tqdm(range(1, tools.COUNTS[category]+1))])

    path = Path(f'cache/nn/output/{category}/{number}.npy')
    if not path.exists():
        path.parent.mkdir(exist_ok=True, parents=True)
        losses, model = load(*MODEL)
        bumps = evaluate(category, number, model)[1]
        sp.save(path, bumps)
    return sp.load(path)

def example(category, number):
    im = tools.image(category, number)
    bumps = bumpmaps(category, number)
    tools.show({'im': im, 'bumps': bumps, 'resid': im - bumps})