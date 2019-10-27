import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FixedLocator
from pylab import fill

def plotMazePolicy(ValueTable, X, Y, U, V, maze, w, h, **kwargs):#show and save default to false. title, colorbar-label, filename
    plt.clf()
    for xy in maze.obstacles:
        U[xy]=0
        V[xy]=0
    mask = np.logical_or(U != 0, V != 0)
    X = X[mask]
    Y = Y[mask]
    U = U[mask]
    V = V[mask]

    fig1, ax1 = plt.subplots()

    if 'title' in kwargs: ax1.set_title(kwargs['title'])

    # Make the arrows, M is the costs
    M = np.array([ValueTable[(i,j)] for i in range(w) for j in range(h)])
    Q = ax1.quiver(X, Y, U, V, M, units='x', pivot='middle', width=0.05,
                    cmap="viridis", scale=1/0.8)

    # Shade and label start cell
    sx,sy=maze.start
    ax1.text(sx+0.2, sy+0.75, 'Start', color="r", fontsize=8)#, transform=ax1.transAxes)
    fill([sx,sx+1,sx+1,sx], [sy,sy,sy+1,sy+1], 'r', alpha=0.2, edgecolor='r')

    # Shade and label goal cell
    ax1.text(w-1+0.2, h-1+0.4, 'Goal', color="g", fontsize=8)#, transform=ax1.transAxes)
    fill([w-1,w,w,w-1], [h-1,h-1,h,h], 'g', alpha=0.2, edgecolor='g')

    # Make obstacles:
    for (x, y) in maze.obstacles: fill([x,x+1,x+1,x], [y,y,y+1,y+1], 'k', alpha=0.2, edgecolor='k')

    for i in range(w):
        for j in range(h):
            if (i,j) not in maze.obstacles:
                ax1.text(0.5+i, 0.1+j, str(round(ValueTable[i, j],2)), horizontalalignment='center',verticalalignment='center')

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

def mazeUVM(πgreedy, V, w, h):# use the max Q value over each action for each state to define the direction
    x,y,u,v = [],[],[],[]
    LV = {}
    mn= min(V.values())
    # print(V.values())
    for k in V: LV[k] = np.log(V[k] + 1 - mn)
    mx, mn = max(LV.values()), min(LV.values())
    # print(LV.values())
    NV = {}
    if mx-mn != 0:
        for k in V: NV[k] = 0.1+0.9*(LV[k]-mn)/(mx-mn)
    else:
        for k in V: NV[k] = 0.0001
    # print(NV)

    for i in range(w):
        xrow, yrow, urow, vrow=[],[],[],[]
        for j in range(h):
            yrow.append(j+0.5)
            xrow.append(i+0.5)
            if πgreedy[(i,j)] == "right":
                urow.append(NV[(i,j)])
                vrow.append(0)
            elif πgreedy[(i,j)] == "up":
                urow.append(0)
                vrow.append(NV[(i,j)])
            elif πgreedy[(i,j)] == "down":
                urow.append(0)
                vrow.append(-NV[(i,j)])
            elif πgreedy[(i,j)] == "left":
                urow.append(-NV[(i,j)])
                vrow.append(0)

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
    # print(u)
    # print(v)
    return x,y,u,v

def maze_record(iteration, tt, V, π, w, h, maze):
    X, Y, u, v = mazeUVM(π, V, w, h)
    fn = "../gif_printer/temp-plots/V"+str(iteration)+".png"
    plotMazePolicy(V, X, Y, u, v, maze, w, h, show=False, filename=fn, title=tt+str(iteration), cbarlbl="Value")
