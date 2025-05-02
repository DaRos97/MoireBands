import numpy as np
from PIL import Image
import os

dirname = "figs_png/"

light = "LH"    #LH,LV,CL

image_name = dirname + "KK_"+light+"_WSe2onWS2_forDario.png"         #experimental data
im  = Image.open(image_name)

pic = np.array(np.asarray(im))
len_e, len_k, z = pic.shape
#Redemension image to wanted values
E_min_fig = -1.4
E_max_fig = -0.2
K_i = -1.1
K_f = -0.1

border_e = 12
border_k = 12
if 0:   #check where the actual image starts
    for i in range(border_e):
        print(i,pic[i,len_k//2])
        print(-i,pic[-i,len_k//2])
#Cut relevant part of the picture and save it for future study
new_pic = pic[border_e:-border_e,border_k:-border_k]
new_image = Image.fromarray(np.uint8(new_pic))
new_imagename = dirname + "cut_KK_"+light+".png"
new_image.save(new_imagename)
os.system("xdg-open "+new_imagename)
























