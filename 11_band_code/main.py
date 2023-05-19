import numpy as np
import sys
import getopt
import os

data_dirname = "../Data/11_bands/"
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "N:",["Path=","LL=","UL=","pts_ps=","mono","gridy=","spread_E="])
    N = 4               #Number of circles of mini-BZ around the central one
    upper_layer = 'WSe2'
    lower_layer = 'WS2'
    pts_ps = 200         #points per step
    Path = 'KGC'
    sbv = [-10,1]
    factor_gridy = 5
    spread_E = 0.005
    spread_K = 0.001      #spread in momentum
    larger_E = 0.2      #in eV. Enlargment of E axis wrt min and max band energies
    plot_mono = False
except:
    print("Error")
    exit()
for opt, arg in opts:
    if opt in ['-N']:
        N = int(arg)
    if opt == '--LL':
        lower_layer = arg
    if opt == '--UL':
        upper_layer = arg
    if opt == '--pts_ps':
        pts_ps = int(arg)
    if opt == '--Path':
        Path = arg

#launch band calculation
args_arpes_bands = (N,upper_layer,lower_layer,pts_ps,Path,sbv,data_dirname)
import arpes_bands
arpes_bands.arpes_bands(args_arpes_bands)

#launch lorentzian spread for K-E plot
args_EK_lorentz = (N,upper_layer,lower_layer,pts_ps,Path,sbv,factor_gridy,spread_E,spread_K,larger_E,data_dirname)
import EK_lorentz
EK_lorentz.EK_lorentz(args_EK_lorentz)

#launch plot for K-E
args_plot_EK = (N,upper_layer,lower_layer,pts_ps,Path,sbv,factor_gridy,spread_E,spread_K,larger_E,plot_mono,data_dirname)
import plot_EK
#plot_EK.plot_EK(args_plot_EK)

#launch lorentzian spread for constant energy maps
args_banana_lorentz = (N,upper_layer,lower_layer,pts_ps,Path,sbv,factor_gridy,spread_E,spread_K,larger_E,plot_mono,data_dirname)
import banana_lorentz
banana_lorentz.banana_lorentz(args_banana_lorentz)

exit()

#launch plot for constant energy maps
com_3b = "python plot_banana.py"
os.system(com_3b)

