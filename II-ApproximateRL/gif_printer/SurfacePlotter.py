from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def surface_plot(xs, ys, zs, filename, xlab, ylab, zlab, title):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_trisurf(ys, xs, zs, cmap=plt.cm.viridis, linewidth=0.2)
	ax.set_xlabel(xlab)
	ax.set_ylabel(ylab)
	ax.set_zlabel(zlab)
	ax.set_title(title)
	plt.savefig('plots/'+filename, dpi=400)
	plt.close()

def surfaceRotation(xs, ys, zs, xlab, ylab, zlab, title):
	i = 0
	for angle in range(0,360,6):

		# Make the plot
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.plot_trisurf(ys, xs, zs, cmap=plt.cm.viridis, linewidth=0.2)

		# Set the angle of the camera
		ax.view_init(30,angle)

		ax.set_xlabel(xlab)
		ax.set_ylabel(ylab)
		ax.set_zlabel(zlab)
		ax.set_title(title)

		# Save it
		filename='../gif_printer/temp-plots/V'+str(i)+'.png'
		plt.savefig(filename, dpi=128)
		plt.gca()
		i+=1
