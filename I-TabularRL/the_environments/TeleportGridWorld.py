class TeleportGridWorld:
    def __init__(self):
        self.w = 5
        self.h = 5
        self.states = [(i,j) for i in range(self.w) for j in range(self.h)]
        self.actions = ["up", "down", "left", "right"]
        self.A = [(1, 4), (1, 0)]
        self.B = [(3, 4), (3, 2)]

    def step(self, state, action):
        if state==self.A[0]: return self.A[1], 10
        if state==self.B[0]: return self.B[1], 5
        x, y = state
        reward = 0
        if action == "up":
            if y == self.h-1: return (x, y), -1
            return (x, y+1), 0
        if action == "down":
            if y == 0: return (x, 0), -1
            return (x, y-1), 0
        if action == "left":
            if x == 0: return (0, y), -1
            return (x-1, y), 0
        if action == "right":
            if x == self.w-1: return (x, y), -1
            return (x+1, y), 0

    def plotgrid(self, ValueTable, filename, policy=None):
        import matplotlib.pyplot as plt
        from pylab import fill
        import numpy as np

        fig, ax = plt.subplots()
        if policy is not None:
            x,y,u,v = [],[],[],[]
            for i in range(self.w):
                xrow, yrow, urow, vrow=[],[],[],[]
                for j in range(self.h):
                    yrow.append(j+0.5)
                    xrow.append(i+0.5)
                    if policy[(i,j)] == "right" :
                        urow.append(1)
                        vrow.append(0)
                    elif policy[(i,j)] == "up":
                        urow.append(0)
                        vrow.append(1)
                    elif policy[(i,j)] == "down":
                        urow.append(0)
                        vrow.append(-1)
                    elif policy[(i,j)] == "left" :
                        urow.append(-1)
                        vrow.append(0)

                x.append(xrow)
                y.append(yrow)
                u.append(urow)
                v.append(vrow)
            X = np.array(x)
            Y = np.array(y)
            U = np.array(u)
            V = np.array(v)
            U[self.A[0]]=0
            V[self.A[0]]=0
            U[self.B[0]]=0
            V[self.B[0]]=0

            mask = np.logical_or(U != 0, V != 0)
            X = X[mask]
            Y = Y[mask]
            U = U[mask]
            V = V[mask]

            # Make the arrows
            Q = ax.quiver(X, Y, U, V, np.zeros((self.w, self.h)), units='x', pivot='middle', cmap="Greys_r", width=0.05, scale=1/0.6)

        for i in range(self.w):
            for j in range(self.h):
                xi, yi = 0.5+i, 0.5+j
                if policy is not None:
                    xi-=0.3
                    yi+=0.3
                ax.text(xi, yi, str(round(ValueTable[i, j], 2)), horizontalalignment='center',verticalalignment='center')
        positions_cols = [(self.A[0], 'g', "A"), (self.A[1], 'r', "A'"), (self.B[0], 'g', "B"), (self.B[1], 'r', "B'")]
        for p, c, l in positions_cols:
            ax.text(p[0]+0.2, p[1]+0.2, l, color=c, fontsize=15)#, transform=ax1.transAxes)
            fill([p[0],p[0]+1,p[0]+1,p[0]], [p[1],p[1],p[1]+1,p[1]+1], c, alpha=0.2, edgecolor=c)
        
        plt.xlim(0,self.w)
        plt.ylim(0,self.h)
        plt.savefig('plots/'+filename, dpi=300)
        plt.show()


