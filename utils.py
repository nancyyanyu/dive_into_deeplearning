import os
import math
import time
import random
import inspect
import requests
import hashlib
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
numpy = lambda x, *args, **kwargs: x.numpy(*args, **kwargs)

def use_svg_display():
    """Use the svg format to display a plot in Jupyter.
    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.
    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.
    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def download(url, folder='../data', sha1_hash=None):
    """Download a file to folder and return the local filepath.
    Defined in :numref:`sec_utils`"""
    if not url.startswith('http'):
        # For back compatability
        url, sha1_hash = DATA_HUB[url]
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, url.split('/')[-1])
    # Check if hit cache
    if os.path.exists(fname) and sha1_hash:
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    # Download
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return 

def extract(filename, folder=None):
    """Extract a zip/tar file into folder.
    Defined in :numref:`sec_utils`"""
    base_dir = os.path.dirname(filename)
    _, ext = os.path.splitext(filename)
    assert ext in ('.zip', '.tar', '.gz'), 'Only support zip/tar files.'
    if ext == '.zip':
        fp = zipfile.ZipFile(filename, 'r')
    else:
        fp = tarfile.open(filename, 'r')
    if folder is None:
        folder = base_dir
    fp.extractall(folder)

def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points.
    Defined in :numref:`sec_calculus`"""

    def has_one_axis(X):  # True if X (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)
    if axes is None:
        axes = plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

class HyperParameters:
    """The base class of hyperparameters."""

    def save_hyperparameters(self, ignore=[]):
        """Defined in :numref:`sec_oo-design`"""
        raise NotImplemented

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.

        Defined in :numref:`sec_utils`"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k: v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)


class ProgressBoard(HyperParameters):
    """The board that plots data points in animation.
    Defined in :numref:`sec_oo-design`"""

    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented

    def draw(self, x, y, label, every_n=1):
        """Defined in :numref:`sec_utils`"""
        Point = collections.namedtuple('Point', ['x', 'y'])
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return

        def mean(x): return sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]),
                          mean([p.y for p in points])))
        points.clear()
        if not self.display:
            return
        use_svg_display()
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(plt.plot([p.x for p in v], [p.y for p in v],
                                          linestyle=ls, color=color)[0])
            labels.append(k)
        axes = self.axes if self.axes else plt.gca()
        if self.xlim:
            axes.set_xlim(self.xlim)
        if self.ylim:
            axes.set_ylim(self.ylim)
        if not self.xlabel:
            self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        display.display(self.fig)
        display.clear_output(wait=True)
        
class DataModule(HyperParameters):
    def __init__(self, root='./data'):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplemented

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
    
    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        # get the (X,y) tensor pairs for training or validation
        tensors = tuple(x[indices] for x in tensors)
        # if training, shuffle the tensors; if validation, no need to shuffle
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tf.data.Dataset.from_tensor_slices(tensors).shuffle(
            buffer_size=shuffle_buffer).batch(self.batch_size)

    
class Module(tf.keras.Model, HyperParameters):
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()
        self.training = None

    def loss(self, y_hat, y):
        raise NotImplemented

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def call(self, X, *args, **kwargs):
        # tf.keras.Model invokes the call method in the built-in __call__ method.
        # redirect call to the forward method, saving its arguments as a class attribute. 
        # We do this to make our code more similar to other framework implementations.
        if kwargs and 'training' in kwargs:
            self.training = kwargs['training']
        return self.forward(X, *args)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, numpy(value), (
            'train_' if train else 'val_') + key, every_n=int(n))

    def training_step(self, batch):
        # self() calls self.__call__
        # self(*batch[:-1]) actually equals to self.forward(batch[:-1])
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplemented
        
        
class Trainer(HyperParameters):
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = len(
            self.val_dataloader) if self.val_dataloader is not None else 0

    def prepare_model(self, model):
        model.trainer = self  # ?
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def prepare_batch(self, batch):
        return batch

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0

        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        self.model.training = True
        for batch in self.train_dataloader:
            with tf.GradientTape() as t:
                loss = self.model.training_step(self.prepare_batch(batch))
            grads = t.gradient(loss, self.model.trainable_variables)
            
            if self.gradient_clip_val > 0:
                grads = self.clip_gradients(self.gradient_clip_val, grads)
            # if self.gradient_clip_val
            self.optim.apply_gradients(
                zip(grads, self.model.trainable_variables))
            self.train_batch_idx += 1

        if self.val_dataloader is None:
            return

        self.model.training = False
        for batch in self.val_dataloader:
            self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx = 0
            
    def clip_gradients(self, grad_clip_val, grads):
        grad_clip_val = tf.constant(grad_clip_val, dtype=tf.float32)
        new_grads = [tf.convert_to_tensor(grad) if isinstance(
            grad, tf.IndexedSlices) else grad for grad in grads]
        norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2) for grad in new_grads)))
        if tf.greater(norm, grad_clip_val):
            for i, grad in enumerate(new_grads):
                new_grads[i] = grad * grad_clip_val / norm
            return new_grads
        return grads
    
            
class FashionMNIST(DataModule):
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        self.train, self.val = tf.keras.datasets.fashion_mnist.load_data()

    def text_labels(self, indices):
        """Return text labels."""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        data = self.train if train else self.val

        def process(X, y): return (tf.expand_dims(
            X, axis=3) / 255, tf.cast(y, dtype=tf.int32))

        def resize_fn(X, y): return (
            tf.image.resize_with_pad(X, *self.resize), y)
        shuffle_buf = len(data[0]) if train else 1
        return tf.data.Dataset.from_tensor_slices(process(*data))\
                              .batch(self.batch_size)\
                              .map(resize_fn)\
                              .shuffle(shuffle_buf)

    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        show_images(tf.squeeze(X), nrows, ncols, titles=labels)

        
class Classifier(Module):
    def validation_step(self, batch):
        y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(y_hat, batch[-1]), train=False)

    def configure_optimizers(self):
        # use a stochastic gradient descent optimizer, operating on minibatches
        return tf.keras.optimizers.SGD(self.lr)

    def accuracy(self, y_hat, y, averaged=True):
        y_hat = tf.reshape(y_hat, (-1, y_hat.shape[-1]))
        preds = tf.cast(tf.argmax(y_hat, axis=1), y.dtype)
        compare = tf.cast(preds == tf.reshape(y, -1), tf.float32)
        return tf.reduce_mean(compare) if averaged else compare
    
    def loss(self,y_hat, y, averaged=True):
        y_hat = tf.reshape(y_hat, (-1, y_hat.shape[-1]))
        y = tf.reshape(y, (-1,))
        # from_logits: Whether `y_pred` is expected to be a logits tensor. 
        # instead of passing softmax probabilities into our new loss function, 
        # we just pass the logits and compute the softmax and its log all at once inside the cross-entropy loss function
        fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return fn(y, y_hat)
    
    def layer_summary(self, X_shape):
        X = tf.random.normal(X_shape)
        for layer in self.net.layers:
            X = layer(X)
            print(layer.__class__.__name__,  'output shape:\t', X.shape)

<<<<<<< HEAD
<<<<<<< HEAD

class LinearRegressionKeras(Module):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        initializer = tf.initializers.RandomNormal(stddev=0.01)
        # only generate a single scalar output, so set the parameter to 1
        self.net = tf.keras.layers.Dense(1, kernel_initializer=initializer)
        
    def forward(self, X):
        # invoke the built-in __call__ method of the predefined layers to compute the outputs.
        return self.net(X)
    
    def loss(self, y_hat, y):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y_hat, y)
    
    def configure_optimizers(self):
        return tf.keras.optimizers.SGD(self.lr)
    
    def get_w_b(self):
        return self.get_weights()[0], self.get_weights()[1]
    
=======
class TimeMachine(DataModule):
    """The Time Machine dataset.
    Defined in :numref:`sec_text-sequence`"""
    def _download(self):
        fname = download(DATA_URL + 'timemachine.txt', self.root,
                             '090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            return f.read()

    def _preprocess(self, text):
        """Defined in :numref:`sec_text-sequence`"""
        return re.sub('[^A-Za-z]+', ' ', text).lower()

    def _tokenize(self, text):
        """Defined in :numref:`sec_text-sequence`"""
        return list(text)

    def build(self, raw_text, vocab=None):
        """Defined in :numref:`sec_text-sequence`"""
        tokens = self._tokenize(self._preprocess(raw_text))
        if vocab is None: vocab = Vocab(tokens)
        corpus = [vocab[token] for token in tokens]
        return corpus, vocab

    def __init__(self, batch_size, num_steps, num_train=10000, num_val=5000):
        """Defined in :numref:`sec_language-model`"""
        super(TimeMachine, self).__init__()
        self.save_hyperparameters()
        corpus, self.vocab = self.build(self._download())
        array = tf.constant([corpus[i:i+num_steps+1]
                            for i in range(len(corpus)-num_steps)])
        self.X, self.Y = array[:,:-1], array[:,1:]

    def get_dataloader(self, train):
        """Defined in :numref:`subsec_partitioning-seqs`"""
        idx = slice(0, self.num_train) if train else slice(
            self.num_train, self.num_train + self.num_val)
        return self.get_tensorloader([self.X, self.Y], train, idx)

>>>>>>> 2624afc... ch9
=======

class LinearRegressionKeras(Module):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        initializer = tf.initializers.RandomNormal(stddev=0.01)
        # only generate a single scalar output, so set the parameter to 1
        self.net = tf.keras.layers.Dense(1, kernel_initializer=initializer)
        
    def forward(self, X):
        # invoke the built-in __call__ method of the predefined layers to compute the outputs.
        return self.net(X)
    
    def loss(self, y_hat, y):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y_hat, y)
    
    def configure_optimizers(self):
        return tf.keras.optimizers.SGD(self.lr)
    
    def get_w_b(self):
        return self.get_weights()[0], self.get_weights()[1]
    
>>>>>>> ae5a7ac... change utils
    
def cpu():
    return tf.device('/CPU:0')

def gpu(i):
    return tf.device(f'/GPU:{i}')

def num_gpus():
    return len(tf.config.experimental.list_physical_devices('GPU'))

def try_gpu(i=0):
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpu():
    return [gpu(i) for i in range(num_gpus())]
