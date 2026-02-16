import numpy as np
import sys,os
from pathlib import Path
import matplotlib.pyplot as plt

if len(sys.argv)!=2:
    print("Usage: py Compare_EDC.py arg1")
    print("arg1: 'G' or 'K'")

momentum_point = sys.argv[1]

data_dn = 'Data/'

data = []
phase = []
for file in Path(data_dn).iterdir():
    file = str(file)[len(data_dn):]
    if file[:5]=='EDC_'+momentum_point:
        phase.append(float(file[9:-4]))
        data.append(np.load(data_dn+file))
        print("Using file: ",file)

fig = plt.figure()
ax = fig.add_subplot()

for i in range(len(data)):
    N_data = data[i].shape[0]
    ax.plot([phase[i],phase[i]],[data[i][N_data//2-1],data[i][N_data//2+1]],color='k',marker='+',lw=0.5)
    ax.scatter(phase[i],data[i][N_data//2],marker='*',color='r')

ax.set_xlabel("Phase "+momentum_point,size=20)
ax.set_ylabel(r"Best distance $\pm0.2°$",size=20)
plt.show()
