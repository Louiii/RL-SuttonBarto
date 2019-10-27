import numpy as np
import sys
sys.path.append('../')

from misc.TilingSutton import *
# Files to plot environment
# from gif_printer.quiverFiles.Quiver3 import *
# from gif_printer.quiverFiles.GridQuiver import *
# Environment files
# from the_environments.ContinuousGrid import GridWorldCts
from the_environments.MountainCar import *
from the_environments.ServersQueue import *

# from misc.mvn import norm_pdf_multivariate

from gif_printer.LegendPlotter import *
from gif_printer.SurfacePlotter import *

from gif_printer.make_gif_from_png import makeGIF