import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os, sys

#sample = '3'
#sample = '11'
sample = sys.argv[1]
print("Fitting mass and offset of sample "+sample)

dirname = "Figs/"
dirname_data = "Data/"
#Open cuted image
image_fn = dirname_data + "S"+sample+"_cuted.npy"
len_e_o, len_k_o, z = np.load(image_fn).shape

#Cut further to fit better
lim_y_d = {'3':450, '11':400}
lim_x_d = {'3':140, '11':150}
pic = np.load(image_fn)[:lim_y_d[sample],lim_x_d[sample]:-lim_x_d[sample],:]
len_e,len_k,z = pic.shape
#Parameters of cuted image and fit image
E_max_cut_d = {'3':-0.5,'11':-0.9}
E_min_cut_d = {'3':-1.7,'11':-2.1}
E_max_cut = E_max_cut_d[sample]
E_min_cut = E_max_cut_d[sample]-(E_max_cut_d[sample]-E_min_cut_d[sample])*len_e/len_e_o
K_cut = 0.6*len_k/len_k_o

if 0:   #Plot fit image
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(np.load(image_fn),cmap='gray')
    plt.xticks([0,len_k_o//2,len_k_o],["{:.2f}".format(-0.6),"0","{:.2f}".format(0.6)])
    plt.yticks([0,len_e_o//2,len_e_o],["{:.2f}".format(E_max_cut),"{:.2f}".format((E_max_cut+E_min_cut_d[sample])/2),"{:.2f}".format(E_min_cut_d[sample])])
    y_off = len_e_o-abs((E_min_cut_d[sample]-E_min_cut)/(E_min_cut_d[sample]-E_max_cut)*len_e_o)
    plt.plot([0,len_k_o],[y_off,y_off],'b')
    plt.subplot(1,2,2)
    plt.imshow(pic,cmap='gray')
    plt.xticks([0,len_k//2,len_k],["{:.2f}".format(-K_cut),"0","{:.2f}".format(K_cut)])
    plt.yticks([0,len_e//2,len_e],["{:.2f}".format(E_max_cut),"{:.2f}".format((E_max_cut+E_min_cut)/2),"{:.2f}".format(E_min_cut)])
    plt.show()
    exit()

#Take for each column the darkest point, above and below a separatrix line b/w the two bands, 
#and fit the points around it with a gaussian --> take max as darkest point
red = np.array([255,0,0,255])
green = np.array([0,255,0,255])
blue = np.array([0,0,255,255])
def inv_gauss(x,a,b,x0,s):
    return -(a*np.exp(-((x-x0)/s)**2)+b)
def find_max(col):
    med = np.argmin(col)
    domain = 10
    in_ = med-domain if med-domain > 0 else 0
    fin_ = med+domain if med+domain < len(col) else -1
    new_arr = col[in_:fin_]
    P0 = [np.max(new_arr)-np.min(new_arr),-np.max(new_arr),np.argmin(new_arr),50]
    try:
        popt,pcov = curve_fit(
            inv_gauss, 
            np.arange(len(new_arr)), new_arr,
            p0 = P0,
#            bounds= ,
            )
        return in_+int(popt[2]) if abs(in_+int(popt[2]))<len(col) else med
    except:
        return med

#Find darkest points in parabula
data_filename = dirname_data + "S"+sample+"_darkest_points.npy"
try:#compute darkest points of the two bands
    data = np.load(data_filename)
except:
    #data array where to save energies of darkest points for the two bands
    data = np.zeros(len_k,dtype=int)
    for x in range(len_k):
        col_up = pic[:,x,0]
        d_up = find_max(col_up)
        data[x] = int(len_e-d_up)
    #save data
    np.save(data_filename,data)

for x in range(len_k):
    d_up = len_e - data[x]
    pic[d_up,x,:] = red

if 0:
    new_im = Image.fromarray(np.uint8(pic))
    plt.figure()
    plt.imshow(new_im)
    plt.xticks([0,len_k//2,len_k],["{:.2f}".format(-K_cut),"0","{:.2f}".format(K_cut)])
    plt.yticks([0,len_e//2,len_e],["{:.2f}".format(E_max_cut),"{:.2f}".format((E_max_cut+E_min_cut)/2),"{:.2f}".format(E_min_cut)])
    plt.show()
    exit()

#fitting of bands with simple model -> parabula with offset
popt_filename = dirname_data + "S"+sample+"_fit_parameters.npy"

k_line = np.linspace(-K_cut,K_cut,len_k)
e_line = np.linspace(E_min_cut,E_max_cut,len_e)
def func(k,m,offset):
    return -k**2/2/m + offset

try:
    popt = np.load(popt_filename)
except:
    offset_d = {'3':-0.67, '11':-1.13}
    Yvals = e_line[data]
    popt,pcov = curve_fit(
            func,
            k_line,Yvals,
            p0=(0.1,offset_d[sample]),
            bounds=([0,-2],[2,-0.5]),
            )
    print(popt)
    #Save result of fitting
    np.save(popt_filename,popt)

fig = plt.figure(figsize=(12,12))
s_ = 20
plt.imshow(pic)

new_k = np.arange(len_k)
new_parabola = len_e*(E_max_cut-func(k_line,*popt))/(E_max_cut-E_min_cut)
plt.plot(new_k,new_parabola,'g')
y_off = len_e-abs((E_min_cut-popt[1])/(E_min_cut-E_max_cut)*len_e)
plt.plot([0,len_k],[y_off,y_off],'b')

plt.xticks([0,len_k//2,len_k],["{:.2f}".format(-K_cut),"0","{:.2f}".format(K_cut)])
plt.yticks([0,len_e//2,len_e],["{:.2f}".format(E_max_cut),"{:.2f}".format((E_max_cut+E_min_cut)/2),"{:.2f}".format(E_min_cut)])
plt.xlabel(r"$K_x\;(\mathring{A}^{-1})$",size=s_)
plt.ylabel(r"$eV$",size=s_)
plt.title("S"+sample+": mass -> "+"{:.5f}".format(popt[0])+", offset -> "+"{:.3f}".format(popt[1])+' eV')
fig.savefig(dirname+'S'+sample+'_fitted_parabula.png')

plt.show()







