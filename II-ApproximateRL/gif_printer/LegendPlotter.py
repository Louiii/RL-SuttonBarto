import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams, cycler
import numpy as np
import json

def multipleCurvesPlot(data, title, xlabel, ylabel, filename, lgplc='upper right',
                        w=10.0, h=4.5, labels=[], log=False, ylims=None):
    labels = list(data.keys()) if len(labels)==0 else labels
    xss = [ data[k][0] for k in labels ]
    yss = [ data[k][1] for k in labels ]
    plt.clf

    f, ax1 = plt.subplots(1, 1)

    f.suptitle(title)
    cmap = plt.cm.PuRd_r
    rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, len(labels))))

    for i in range(len(labels)):
        ax1.plot(xss[i], yss[i], color=cmap((i)*(0.9*1/len(labels))), lw=1.0, label=labels[i])
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel(xlabel)

    ax1.legend(loc=lgplc, fancybox=True, shadow=True)
    if log: plt.yscale("log")
    if ylims is not None: plt.ylim(ylims)
    f.set_size_inches(w,h)
    plt.plot()
    plt.savefig("plots/"+filename+'.png', dpi=400)#, bbox_inches = 'tight')
    # plt.show()
    plt.close()

def plot_4_figs(data, title, filename):
    """ data is a list containing dictionary's for each plot:
    key = labels for legend
    value = (xs, ys, ylqs, yuqs) where xs, ys, yuqs, ylqs = list of floats
    there are keys "xlab", "ylab", "title"
    """
    fig = plt.figure()

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    axes = [ax1, ax2, ax3, ax4]

    for i in range(4):
        ax = axes[i]
        d  = data[i]
        ax.title.set_text(d["title"])
        ax.set_ylabel(d["ylab"])
        ax.set_xlabel(d["xlab"])
        ax.set_ylim(d["ylims"])

        labels = list( set(d.keys() ) - set(["xlab", "ylab", "title", "ylims"]) )
        cmap = plt.cm.cool
        rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, len(labels))))
        mn, mx = float('inf'), -float('inf')
        for j in range(len(labels)):
            l = labels[j]
            xs = d[l][0]
            ylqs, ys, yuqs = d[l][2], d[l][1], d[l][3]
            # lqs, uqs = np.array(ylqs) + np.array(ys), np.array(yuqs) + np.array(ys)
            # mn, mx = min([mn]+list(lqs)), max([mx]+list(uqs))
            ax.errorbar(xs, ys, yerr=[ylqs, yuqs], fmt='--o', color=cmap(j/len(labels)), lw=1.0, label=l)
        # delta = (mx-mn)*0.1
        # ax.set_ylim([mn-delta, mx+delta])

        ax.legend(loc='upper left', fancybox=True, shadow=True)

    fig.suptitle(title, fontsize=16)
    fig.set_size_inches(10,10)
    plt.savefig(filename, dpi=400)
    plt.show()

def export_dataset(data, name):
    with open('datasets/'+name+'.json', 'w') as outfile:
        json.dump(data, outfile)

def load_dataset(name):
    json_file = open('datasets/'+name+'.json')
    json_str = json_file.read()
    return json.loads(json_str)
