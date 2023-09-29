import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

dirname = "figs_png/"
dirname_data = "data_fits/"
#Open image and take pixels
light = "CL"    #LH,LV,CL
filename = dirname + "cut_KK_"+light+".png"
im  = Image.open(filename)

#to array
pic = np.array(np.asarray(im))
len_e,len_k,z = pic.shape

#Take for each column the darkest point, above and below a separatrix line b/w the two bands, 
#and fit the points around it with a gaussian --> take max as darkest point
red = np.array([255,0,0,255])
green = np.array([0,255,0,255])
blue = np.array([0,0,255,255])
def inv_gauss(x,a,b,x0,s):
    return -(a*np.exp(-((x-x0)/s)**2)+b)
def find_max(col):
    med = np.argmin(col)
    domain = 50
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
    print(popt)
    xx = np.linspace(0,len(new_arr),1000)
    plt.plot(xx,inv_gauss(xx,*popt),'k-')
    plt.plot(np.arange(len(new_arr)),new_arr,'r*')
    plt.show()

if 0:#print border between the two bands
    for x in range(len_k):
        border = len_e-1-int(x*1.2)
        pic[border,x] = green
    for e in range(len_e):
        pic[e,len_k//2-120] = red
    new_image = Image.fromarray(np.uint8(pic))
    new_imagename = "temp.png"
    new_image.save(new_imagename)
    os.system("xdg-open "+new_imagename)
    exit()

K_lim = len_k//2 - 120          #limit where to consider the band

new_filename = dirname + "K_extracted_points_"+light+".png"
data_filename = dirname_data + "K_extracted_points_"+light+".npy"
try:#compute darkest points of the upper band
    im  = Image.open(filename)
    data = np.load(data_filename)
except:
    #data array where to save energies of darkest points for the upper band
    data = np.zeros(K_lim,dtype=int)
    for x in range(K_lim):
        #upper band
        border = len_e-1-int(x*1.2)
        col_up = pic[:border,x,0]
        d_up = find_max(col_up)
#        d_up = np.argmin(col_up)
        pic[d_up,x,:] = red
        data[x] = int(len_e-d_up)
    #save data
    np.save(data_filename,data)
    #save new image
    new_im = Image.fromarray(np.uint8(pic))
    new_im.save(new_filename)

if 0:   #plot the extracted points
    os.system("xdg-open "+new_filename)
    exit()

#fitting of band with simple model 
#Parameters of image
#Redemension image to wanted values
E_min_fig = -1.4
E_max_fig = -0.2
K_i_image = -1.1
K_f_image = -0.1
#New K_f
K_f = K_i_image + (K_f_image-K_i_image)*K_lim/len_k       #value of K where we put the top of the band (K_lim)
#Then, take a different K_i to consider only the parabolic part of the band
K_i = -0.8
K_i_pix = int(len_k*(K_i-K_i_image)/(K_f_image-K_i_image))
#Now, center k to top of band so that K_f = 0, we just need the distance b/w K_f and K_i
K_cut = K_i-K_f     #negative

k_line = np.linspace(K_cut,0,K_lim-K_i_pix)
e_line = np.linspace(E_min_fig,E_max_fig,len_e)
X = k_line
Y = e_line[data[K_i_pix:]]
if 0:   #plot considered points
    plt.plot(X,Y)
    plt.show()
    exit()

def func1(k,m1,mu):
    return -k**2/2/m1 + mu
#Combined fit
popt,pcov = curve_fit(
        func1,
        X,Y,
        p0=(1,-0.2),
        bounds=([0.01,-1],[5,-0.05]),
        )
print(popt)

if 0:#plot
    plt.figure()
    for j in range(len(k_line)):
        plt.scatter(X[j],Y[j],color='r')
    plt.plot(k_line,func1(k_line,*popt),'k-')
    plt.xlim(k_line[0],k_line[-1])
    plt.ylim(-1.4,-0.2)
    plt.show()
    exit()
if 0:#png image
    k_all = np.linspace(K_i-K_f,0,K_lim-K_i_pix)
    off = 0
    for x in range(off,len(k_all)-off):
        ind_e = len_e - int((E_min_fig-func1(k_all[x],*popt))/(E_min_fig-E_max_fig)*len_e)
        pic[ind_e,K_i_pix+x,:] = red
    new_image = Image.fromarray(np.uint8(pic))
    new_imagename = "temp.png"
    new_image.save(new_imagename)
    os.system("xdg-open "+new_imagename)
    exit()

#Save result of fitting
popt_filename = dirname_data + "K_popt_interlayer.npy"
np.save(popt_filename,popt)





