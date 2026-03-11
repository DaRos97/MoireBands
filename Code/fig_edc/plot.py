""" Here I plot the EDC results. """

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,              # Use LaTeX for all text
    "font.family": "serif",           # Set font family
    "font.serif": ["Computer Modern"], # Default LaTeX font
    "text.latex.preamble": r"\usepackage{amsmath}",
})
cmap = 'cividis'
colG = 'red'
ecolG = 'darkred'
colK = 'lime'
ecolK = 'green'

""" Data """
dataG = np.load('Inputs/data_EDC.npy')
dataK = np.load('Inputs/data_EDC_k.npy')
listPhiK = np.linspace(-180,180,360,endpoint=False)
for i in range(dataK.shape[0]):
    dataK[i,0] = listPhiK[int(dataK[i,0])]
    if dataK[i,0]< -50:
        dataK[i,0] += 360

indSpecialG = 1150
indSpecialK = 70

fig = plt.figure(figsize=(7,3))
ax = fig.add_subplot()

""" Plotted data """
ax.scatter(
    dataG[:,2]/np.pi*180,
    dataG[:,3],
    color=colG,
    marker='o',
    zorder=9
)
ax.scatter(
    dataK[:,0],
    dataK[:,1],
    color=colK,
    marker='o',
    zorder=9
)

ax.scatter(
    dataG[indSpecialG,2]/np.pi*180,
    dataG[indSpecialG,3],
    color=colG,
    edgecolor=ecolG,
    lw=1.5,
    s=100,
    zorder=10,
    marker='*'
)
ax.scatter(
    dataK[indSpecialK,0],
    dataK[indSpecialK,1],
    color=colK,
    edgecolor=ecolK,
    lw=1.5,
    s=100,
    zorder=10,
    marker='*'
)

for i in range(6):
    ax.axvline(
        60*i,
        color=ecolK if i%2 else ecolG,
        ls=(0,(4,2)),
        lw=0.5
    )

""" Background and colorbar """
xmin,xmax = ax.get_xlim()
ymin,ymax = (0,0.02)
xline = np.linspace(xmin,xmax,200)
yline = np.linspace(ymin,ymax,120)
X,Y = np.meshgrid(xline,yline)
pm = ax.pcolormesh(
    xline,
    yline,
    12*Y*abs(np.cos((X%360)/180*np.pi)),
    cmap='cividis',
    zorder=-1,
    rasterized=True
)

cbar = plt.colorbar(
    pm,
    ax=ax,
    fraction=0.1,
    aspect=10,
    pad=0.02,
)
cbar.ax.axhline(
    12*dataG[indSpecialG,3]*abs(np.cos((dataG[indSpecialG,2]%360)/180*np.pi)),
    color=colG,
    lw=2
)
cbar.ax.axhline(
    12*dataK[indSpecialK,1]*abs(np.cos((dataK[indSpecialK,0]%360)/180*np.pi)),
    color=colK,
    lw=2
)

ax.set_ylim(ymin,ymax)
ax.set_xlim(xmin,xmax)

""" Labels and axes """
s_ = 12
ax.set_xlabel(r'$\phi$ [$^\circ$]',size=s_)
ax.set_ylabel(r'$V$ [eV]',size=s_)
cbar.set_label(r'$\Delta V$ [eV]',size=s_)
ax.set_xticks(
    [60*i for i in range(6)],
    [int(60*i) for i in range(6)],
    size=s_
)
ax.tick_params(
    labelsize=s_
)

plt.subplots_adjust(
    top=0.976,
    right=0.962,
    bottom=0.17,
    left=0.11
)

plt.show()

if 1:
    fig.savefig('Data/Fig_EDC.pdf')
