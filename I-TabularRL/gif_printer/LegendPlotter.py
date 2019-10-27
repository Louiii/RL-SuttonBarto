import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams, cycler
import numpy as np
import json

def multipleCurvesPlot(data, title, xlabel, ylabel, filename, lgplc='upper right',
                        w=10.0, h=4.5, xlims=None, ylims=None, labels=[], xlog=None, xticks=None, leg=True):
    labels = list(data.keys()) if len(labels)==0 else labels
    xss = [ data[k][0] for k in labels ]
    yss = [ data[k][1] for k in labels ]
    plt.clf

    f, ax = plt.subplots(1, 1)

    f.suptitle(title)
    cmap = plt.cm.PuRd_r
    rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, len(labels))))

    for i in range(len(labels)):
        ax.plot(xss[i], yss[i], color=cmap((i)*(0.9*1/len(labels))), lw=1.0, label=labels[i])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if leg: ax.legend(loc=lgplc, fancybox=True, shadow=True)
    if ylims is not None: plt.ylim(ylims)
    if xlims is not None: plt.xlim(xlims)
    if xticks is not None:
        plt.xticks(xticks[0], xticks[1])  # Set locations and labels
        plt.tick_params(axis='x', which='minor', bottom=False)
    if xlog is not None: 
        plt.xscale('log')
        plt.xticks(xlog[0], xlog[1])  # Set locations and labels
        plt.tick_params(axis='x', which='minor', bottom=False)
    f.set_size_inches(w,h)
    plt.plot()
    plt.savefig("plots/"+filename+'.png', dpi=400)#, bbox_inches = 'tight')
    plt.show()
    plt.close()

def export_dataset(data, name):
    with open('datasets/'+name+'.json', 'w') as outfile:
        json.dump(data, outfile)

def load_dataset(name):
    json_file = open('datasets/'+name+'.json')
    json_str = json_file.read()
    return json.loads(json_str)
