import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

dirname = "Figs/"

#sample = '3'
sample = '11'

image_name = dirname + "S"+sample+"_KGK.png"         #experimental data
im  = Image.open(image_name)

pic = np.array(np.asarray(im))
if 0: #extract coordinates of just image without axis
    for y in range(pic.shape[0]):
        for x in range(pic.shape[1]):
            pix = pic[-y,-x]
            if pix[-1] == 0:
                continue
            if pix[0] == 0:
                continue
            print(x,y,pix)
            input()
X_i = {'3':695, '11':808
        }
X_f = {'3':pic.shape[1]-475, '11':pic.shape[1]-595
        }
Y_i = {'3':108, '11':90
        }
Y_f = {'3':pic.shape[0]-327, '11':pic.shape[0]-305
        }

x_i = X_i[sample]
y_i = Y_i[sample]
x_f = X_f[sample]
y_f = Y_f[sample]
pic = np.array(np.asarray(im)[y_i:y_f,x_i:x_f])

if 0: #cut old image to get just new image without axes
    fig = plt.figure(figsize=(12,10))
    plt.imshow(pic,cmap='gray')
    plt.axis('off')
    plt.show()
    exit()

#Redemension image to wanted values
E_max_fig = 0
E_min_fig_d = {'3':-2.5, '11':-3.5}
E_min_fig = E_min_fig_d[sample]
K_fig = 1

E_max_cut_d = {'3':-0.5,'11':-0.9}
E_min_cut_d = {'3':-1.7,'11':-2.1}
E_max_cut = E_max_cut_d[sample]
E_min_cut = E_min_cut_d[sample]
K_cut = 0.6
#
ind_e_M = int(abs((y_f-y_i)/(E_max_fig-E_min_fig)*E_max_cut))
ind_e_m = int(abs((y_f-y_i)/(E_max_fig-E_min_fig)*E_min_cut))

ind_k_m = int(abs((x_f-x_i)/2*(1-K_cut/K_fig)))
ind_k_M = int(abs((x_f-x_i)/2*(1+K_cut/K_fig)))

if 1: #frame selected region
    green = np.array([0,255,0,255])
    for y in range(ind_e_m-ind_e_M):
        pic[ind_e_M+y,ind_k_m] = green
        pic[ind_e_M+y,ind_k_M] = green
    for x in range(ind_k_M-ind_k_m):
        pic[ind_e_M,ind_k_m+x] = green
        pic[ind_e_m,ind_k_m+x] = green
    plt.imshow(pic,cmap='gray')
    plt.show()
    exit()

#Cut relevant part of the picture and save it for future study
new_pic = pic[ind_e_M:ind_e_m,ind_k_m:ind_k_M]
new_image = Image.fromarray(np.uint8(new_pic))
new_imagename = dirname + "S"+sample+"_cuted.png"
new_image.save(new_imagename)
os.system("xdg-open "+new_imagename)

























