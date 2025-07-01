import sys,os
import numpy as np
import scipy
cwd = os.getcwd()
master_folder = cwd[:35]+'Code/'
print(master_folder)
sys.path.insert(1, master_folder)
import CORE_functions as cfs
from pathlib import Path
import matplotlib.pyplot as plt

from PIL import Image

figS3 = '../Code/4_newMoire/Inputs/S3_KGK.png'
figS11 = '../Code/4_newMoire/Inputs/S11_KGK.png'

kList = cfs.get_kList('Kp-G-K',100)
K0 = np.linalg.norm(kList[0])   #val of |K|

pic_rawS11 = np.array(np.asarray(Image.open(figS11)))
totPe11,totPk11,_ = pic_rawS11.shape
E_max11, E_min11, pKi11, pKf11, pEmax11, pEmin11 = cfs.dic_pars_samples['S11']
pK011 = int((pKf11+pKi11)/2)   #pixel of middle -> k=0
pKF11 = int((pKf11-pK011)*K0+pK011)   #pixel of k=|K|
pKI11 = 2*pK011-pKF11             #pixel of k=-|K|
picS11 = pic_rawS11[pEmax11:pEmin11,pKI11:pKF11]
picS11[:,:(pK011-pKI11),:3] = 1

picS11[:,(pK011-pKI11):,:3] = (picS11[:,(pK011-pKI11):,:3])**(1.1)

#S3
pic_rawS3 = np.array(np.asarray(Image.open(figS3)))
totPe3,totPk3,_ = pic_rawS3.shape
E_max3, E_min3, pKi3, pKf3, pEmax3, pEmin3 = cfs.dic_pars_samples['S3']
pK03 = int((pKf3+pKi3)/2)   #pixel of middle -> k=0
pKF3 = int((pKf3-pK03)*K0+pK03)   #pixel of k=|K|
pKI3 = 2*pK03-pKF3             #pixel of k=-|K|

picS3 = pic_rawS3[pEmax3:pEmin3,pKI3:pKF3]
picS3[:,(pK03-pKI3):,3] = 0
picS3[:,:(pK03-pKI3),:3] = (picS3[:,:(pK03-pKI3),:3])**(1.1)

e3,k3,_ = picS3.shape
e11,k11,_ = picS11.shape

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot()
ax.imshow(picS11,
          extent=[0,k11,e11,0],
          zorder=-1
         )

k30 = int(k11/2-k3/2)
k3f = k30 + k3

et = int(e11/3.5*0.47)
eb = int(et + e11/3.5*2.5)

ax.imshow(picS3,
          extent=[k30,k3f,eb,et],
          zorder = 2,
         )

yKl = 480
yKu = 450
ax.plot([k30,k11],[yKl,yKl],color='r',zorder=10,lw=1,alpha=0.2)
ax.plot([k30,k11],[yKu,yKu],color='r',zorder=10,lw=1,alpha=0.2)

ax.set_ylim(e11,0)
ax.set_xlim(k30,k11)


if 1:
    fig.savefig("ComparisonS3_S11_0,47_saturated.png")





plt.show()
