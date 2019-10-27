import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FixedLocator
from pylab import fill

def plotPolicy(ValueTable, X, Y, U, V, M, w, h, **kwargs):#show and save default to false. title, colorbar-label, filename
    plt.clf()

    mask = np.logical_or(U != 0, V != 0)
    X = X[mask]
    Y = Y[mask]
    U = U[mask]
    V = V[mask]

    fig1, ax1 = plt.subplots()

    if 'title' in kwargs:
        ax1.set_title(kwargs['title'])

    # Make the arrows
    Q = ax1.quiver(X, Y, U, V, M, units='x', pivot='middle', width=0.05,
                    cmap="autumn_r", scale=1/0.8)

    # Shade and label goal cell
    ax1.text(w-1+0.2, h-1+0.4, 'Goal', color="g", fontsize=15)#, transform=ax1.transAxes)
    fill([w-1,w,w,w-1], [h-1,h-1,h,h], 'g', alpha=0.2, edgecolor='g')

    for i in range(w):
        for j in range(h):
            ax1.text(0.5+i, 0.25+j, str(int(round(ValueTable[i, j]))), horizontalalignment='center',verticalalignment='center')

    if "cbarlbl" in kwargs:
        # Create colorbar
        cbar = ax1.figure.colorbar(Q, ax=ax1)#, **cbar_kw)

        t = kwargs["cbarlbl"] # label colorbar
        cbar.ax.set_ylabel(t, rotation=-90, va="bottom")

    # make grid lines on center
    plt.xticks([0.5+i for i in range(w)], [i for i in range(w)])
    plt.yticks([0.5+i for i in range(h)], [i for i in range(h)])
    plt.xlim(0,w)
    plt.ylim(0,h)
    # make grid
    minor_locator1 = AutoMinorLocator(2)
    minor_locator2 = FixedLocator([j for j in range(h)])
    plt.gca().xaxis.set_minor_locator(minor_locator1)
    plt.gca().yaxis.set_minor_locator(minor_locator2)
    plt.grid(which='minor')

    plt.xlabel("cell x coord.")
    plt.ylabel("cell y coord.")
    # plt.tight_layout()

    if 'filename' in kwargs:
        plt.savefig(kwargs['filename'], dpi=200)
    if 'show' in kwargs:
        if kwargs['show']:
            plt.show()

def makeUVM(πgreedy, w, h):# use the max Q value over each action for each state to define the direction
    x,y,u,v = [],[],[],[]

    for i in range(w):
        xrow, yrow, urow, vrow=[],[],[],[]
        for j in range(h):
            yrow.append(j+0.5)
            xrow.append(i+0.5)
            if πgreedy[(i,j)] == "right" :
                urow.append(1)
                vrow.append(0)
            else:
                urow.append(0)
                vrow.append(1)

        x.append(xrow)
        y.append(yrow)
        u.append(urow)
        v.append(vrow)
    x = np.array(x)
    y = np.array(y)
    u = np.array(u)
    v = np.array(v)
    u[w-1, h-1]=0
    v[w-1, h-1]=0
    return x,y,u,v

def record(iteration, tt, V, π, w, h, costs):
    X, Y, u, v = makeUVM(π, w, h)
    fn = "../gif_printer/temp-plots/V"+str(iteration)+".png"
    plotPolicy(V, X, Y, u, v, costs, w, h, show=False, filename=fn, title=tt+str(iteration), cbarlbl="Costs")
