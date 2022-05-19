# That's an impressive list of imports.
import torch
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist
import ipdb
import argparse
import os

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
#from sklearn.manifold.t_sne import (_joint_probabilities,
#                                    _kl_divergence)

# Random state.
RS = 20150101

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

parser = argparse.ArgumentParser(description='The t-SNE')
parser.add_argument('--plot_dir', default='', type=str, help='directory of data to be plotted')
parser.add_argument('--save_dir', default='', type=str, help='directory to save plotted figure')
parser.add_argument('--fig_name', default='', type=str, help='name of plotted figure')
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

def scatter(x, colors):
    # We choose a color palette with seaborn.
    '''For SSL CIFAR-10
    if colors.max() == 1:
        tmp = sns.hls_palette(8, l=.5, s=1)
        palette = np.array([tmp[0], tmp[5]])
    elif colors.max() == 9:
        palette = np.array(sns.color_palette("hls", 10))
    '''
    #tmp = sns.color_palette("Reds_r", n_colors=31)
    #tmp.extend(sns.color_palette("Blues_r", n_colors=31))
    #palette = np.array(tmp)
    
    palette = np.array(sns.color_palette("hls", colors.astype(np.int).max()+1))
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    plt.rcParams['figure.dpi'] = 300
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=5, c=palette[colors.astype(np.int)])
    #sc1 = ax.scatter(x[:2817,0], x[:2817,1], lw=0, s=5, c=palette[colors[:2817].astype(np.int)])#498
    #sc2 = ax.scatter(x[2817:,0], x[2817:,1], lw=0.3, s=5, c=palette[colors[2817:].astype(np.int)], marker='x')
    #max_value = int(np.abs(x).max() * 1.2)
    #plt.xlim(-max_value, max_value)
    #plt.ylim(-max_value, max_value)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

data = torch.load(os.path.join(args.plot_dir, 'feature_label_lists.pth.tar'))
X = data['features']
Y = data['labels']
proj = TSNE(random_state=RS).fit_transform(X)
scatter(proj, Y)
#plt.show()
plt.savefig(os.path.join(args.save_dir, args.fig_name))