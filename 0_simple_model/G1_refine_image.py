import numpy as np
from PIL import Image
import os

dirname = "figs_png/"

image_name = dirname + "KGK_WSe2onWS2_v2.png"         #experimental data
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
x_i = 441
y_i = 110
x_f = pic.shape[1]-221
y_f = pic.shape[0]-327
if 0: #cut old image to get just new image without axes
    new_pic = np.array(np.asarray(im)[y_i:y_f,x_i:x_f])
    #save cut image
    new_image = Image.fromarray(np.uint8(new_pic))
    new_imagename = dirname + "full_KGK_no_axis.png"
    new_image.save(new_imagename)
    os.system("xdg-open "+new_imagename)

#Redemension image to wanted values
E_min_fig = -2.5
E_max_fig = 0
a_WSe2 = 3.32 
K_fig = 4*np.pi/3/a_WSe2        #distance from Gamma to K

E_min_cut = -1.7
E_max_cut = -0.5
K_cut = 0.5
#
offset_k = 26 #additional pixels because the picture is not exactly KGK, goes a bit further

ind_e_M = y_i+int(abs((y_f-y_i)/(E_max_fig-E_min_fig)*E_max_cut))
ind_e_m = y_i+int(abs((y_f-y_i)/(E_max_fig-E_min_fig)*E_min_cut))

ind_k_m = x_i+int(abs((x_f-x_i)/2*(1-K_cut/K_fig))+offset_k)
ind_k_M = x_i+int(abs((x_f-x_i)/2*(1+K_cut/K_fig))-offset_k)

if 0: #frame selected region
    green = np.array([0,255,0,255])
    for y in range(ind_e_m-ind_e_M):
        pic[ind_e_M+y,ind_k_m] = green
        pic[ind_e_M+y,ind_k_M] = green
    for x in range(ind_k_M-ind_k_m):
        pic[ind_e_M,ind_k_m+x] = green
        pic[ind_e_m,ind_k_m+x] = green
    new_image = Image.fromarray(np.uint8(pic))
    new_imagename = "temp.png"
    new_image.save(new_imagename)
    os.system("xdg-open "+new_imagename)

#Cut relevant part of the picture and save it for future study
new_pic = pic[ind_e_M:ind_e_m,ind_k_m:ind_k_M]
new_image = Image.fromarray(np.uint8(new_pic))
new_imagename = dirname + "cut_KGK.png"
new_image.save(new_imagename)
os.system("xdg-open "+new_imagename)

























