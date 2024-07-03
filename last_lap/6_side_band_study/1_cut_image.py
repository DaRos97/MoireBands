import numpy as np
from PIL import Image
import os,sys
import matplotlib.pyplot as plt

dirname = "Figs/"
data_dn = "Data/"

#sample = '3'
#sample = '11'
sample = sys.argv[1]
print("Cutting image of sample "+sample)

image_name = dirname + "S"+sample+"_KGK.png"         #experimental data
im  = Image.open(image_name)

original_pic = np.array(np.asarray(im))

X_i = {'3':695, '11':808}
X_f = {'3':original_pic.shape[1]-475, '11':original_pic.shape[1]-595}
Y_i = {'3':108, '11':90}
Y_f = {'3':original_pic.shape[0]-327, '11':original_pic.shape[0]-305}

x_i = X_i[sample]
y_i = Y_i[sample]
x_f = X_f[sample]
y_f = Y_f[sample]
pic = np.array(np.asarray(im)[y_i:y_f,x_i:x_f])
len_e,len_k,z = pic.shape

if 0: #cut old image to get just new image without axes
    fig = plt.figure(figsize=(12,10))
    plt.imshow(pic,cmap='gray')
    plt.axis('off')
    plt.show()
    exit()

#Values of axis for X_i,X_f and Y_i,Y_f
E_max_fig = 0
E_min_fig_d = {'3':-2.5, '11':-3.5}
E_min_fig = E_min_fig_d[sample]
K_fig = 1
#Redemension image to wanted values
E_max_cut_d = {'3':-0.5,'11':-0.9}
E_min_cut_d = {'3':-1.7,'11':-2.1}
E_max_cut = E_max_cut_d[sample]
E_min_cut = E_min_cut_d[sample]
K_cut = 0.6
#
ind_e_M = int((E_max_cut-E_max_fig)/(E_min_fig-E_max_fig)*len_e)
ind_e_m = int((E_min_cut-E_max_fig)/(E_min_fig-E_max_fig)*len_e)

ind_k = int((K_fig-K_cut)/2/K_fig*len_k)

#Cut relevant part of the picture and save it for future study
new_pic = pic[ind_e_M:ind_e_m,ind_k:len_k-ind_k]
new_image = Image.fromarray(np.uint8(new_pic))
new_imagename = dirname + "S"+sample+"_cuted.png"
data_imagename = data_dn + "S"+sample+"_cuted.npy"
np.save(data_imagename, new_pic)
new_image.save(new_imagename)

fig = plt.figure(figsize=(15,8))
plt.subplot(1,3,1)
plt.imshow(original_pic)
plt.axis('off')

plt.subplot(1,3,2)
green = np.array([0,255,0,255])
for y in range(ind_e_m-ind_e_M):
    pic[ind_e_M+y,ind_k] = green
    pic[ind_e_M+y,len_k-ind_k] = green
for x in range(len_k-2*ind_k):
    pic[ind_e_M,ind_k+x] = green
    pic[ind_e_m,ind_k+x] = green
plt.imshow(pic,cmap='gray')
plt.xticks([0,len_k//2,len_k],["{:.2f}".format(-K_fig),"0","{:.2f}".format(K_fig)])
plt.yticks([0,len_e//2,len_e],["{:.2f}".format(E_max_fig),"{:.2f}".format((E_max_fig+E_min_fig)/2),"{:.2f}".format(E_min_fig)])

plt.subplot(1,3,3)
plt.imshow(new_pic)
len_e,len_k,z = new_pic.shape
plt.xticks([0,len_k//2,len_k],["{:.2f}".format(-K_cut),"0","{:.2f}".format(K_cut)])
plt.yticks([0,len_e//2,len_e],["{:.2f}".format(E_max_cut),"{:.2f}".format((E_max_cut+E_min_cut)/2),"{:.2f}".format(E_min_cut)])

plt.show()
#os.system("xdg-open "+new_imagename)

























