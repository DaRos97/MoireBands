import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

dirname = "input_figures/"
dirname_data = "data_fits/"
#Open image and take pixels
filename = dirname + "cut_KGK_v1.png"

im  = Image.open(filename)

#to array
pic = np.array(np.asarray(im))
len_e,len_k,z = pic.shape

#data array where to save energies of darkest points for the two bands
data = np.zeros((2,len_k),dtype=int)

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
    print(popt)
    xx = np.linspace(0,len(new_arr),1000)
    plt.plot(xx,inv_gauss(xx,*popt),'k-')
    plt.plot(np.arange(len(new_arr)),new_arr,'r*')
    plt.show()

if 0:#print border between the two bands
    for x in range(len_k):
        border = len_e//2+(x-len_k//2)**2//600 #400
        pic[border,x] = green
    plt.imshow(pic,cmap='gray')
    plt.show()
    exit()

new_filename = dirname + "G_extracted_points_v1.png"
data_filename = dirname_data + "G_extracted_points_v1.npy"
try:#compute darkest points of the two bands
    im  = Image.open(filename)
    data = np.load(data_filename)
except:
    for x in range(len_k):
        #upper band
        border = len_e//2+(x-len_k//2)**2//600  #400
        col_up = pic[:border,x,0]
        d_up = find_max(col_up)
#        d_up = np.argmin(col_up)
        pic[d_up,x,:] = red
        data[0,x] = int(len_e-d_up)
        #
        #lower band
        col_low = pic[border:,x,0]
        d_low = find_max(col_low)
#        d_low = np.argmin(col_low)
        pic[border+d_low,x,:] = blue
        data[1,x] = int(len_e-(border+d_low))
    #save data
    np.save(data_filename,data)
    #save new image
    new_im = Image.fromarray(np.uint8(pic))
    new_im.save(new_filename)

#fitting of bands with simple model 
popt_filename = dirname_data + "G_popt_interlayer_v1.npy"
#Parameters of image
E_min_cut = -2.2#-1.7
E_max_cut = -0.9#-0.5
K_cut = 0.5
r = 0.1         #removed k space from fit
k_line = np.linspace(-K_cut+r,K_cut-r,int(len_k*(1-r/K_cut)))
e_line = np.linspace(E_min_cut,E_max_cut,len_e)
def func1(k,a,b,c,m1,m2,mu):#up
    alpha = k**2/2/m1+k**2/2/m2+c
    beta = k**2/2/m1*(k**2/2/m2+c) - a**2*(1-b*k**2)**2
    step = np.sqrt(alpha**2-4*beta)
    return (-alpha+step)/2 + mu
def func2(k,a,b,c,m1,m2,mu):#low
    alpha = k**2/2/m1+k**2/2/m2+c
    beta = k**2/2/m1*(k**2/2/m2+c) - a**2*(1-b*k**2)**2
    step = np.sqrt(alpha**2-4*beta)
    return (-alpha-step)/2 + mu
def combinedFunc(combK,a,b,c,m1,m2,mu):
    extract1 = combK[:len(k_line)]
    extract2 = combK[len(k_line):]
    res1 = func1(extract1,a,b,c,m1,m2,mu)
    res2 = func2(extract2,a,b,c,m1,m2,mu)
    return np.append(res1, res2)
try:
    popt = np.load(popt_filename)
except:
    comboX = np.append(k_line,k_line)   #k-data for the two parabolas
    rem_e = int(len_k/2/K_cut*r)+1      #removed energy data because of r
    comboY = np.append(e_line[data[0,rem_e:-rem_e]],e_line[data[1,rem_e:-rem_e]])  #e-data for the two parabolas
    #Combined fit
    popt,pcov = curve_fit(
            combinedFunc,
            comboX,comboY,
            p0=(-0.1,-1,0.7,0.1,0.1,-0.5),
            bounds=([-10,-10,0.1,0.01,0.01,-2],[10,10,2.7,5,5,2]),
            )
    print(popt)
    #Save result of fitting
    np.save(popt_filename,popt)

if 1:#plot
    plt.figure(figsize=(12,12))
    s_ = 20
    plt.imshow(pic)
    new_k = np.arange(r/(2*K_cut)*len_k,len_k-r/(2*K_cut)*len_k-1)
    new_ul = len_e*(E_max_cut-func1(k_line,*popt))/(E_max_cut-E_min_cut)
    new_ll = len_e*(E_max_cut-func2(k_line,*popt))/(E_max_cut-E_min_cut)
    plt.plot(new_k,new_ul,'r')
    plt.plot(new_k,new_ll,'b')
    plt.xticks([0,len_k//2,len_k],["{:.1f}".format(K_cut),"0","{:.1f}".format(K_cut)])
    plt.yticks([0,len_e//2,len_e],["{:.1f}".format(E_max_cut),"{:.1f}".format((E_max_cut+E_min_cut)/2),"{:.1f}".format(E_min_cut)])
    plt.xlabel(r"$K_x\;(\mathring{A}^{-1})$",size=s_)
    plt.ylabel(r"$eV$",size=s_)
    plt.show()
if 0:#png image
    k_all = np.linspace(-K_cut,K_cut,len_k)
    off = 20
    for x in range(off,len_k-off):
        #upper band
        ind_e_up = len_e - int((E_min_cut-func1(k_all[x],*popt))/(E_min_cut-E_max_cut)*len_e)
        ind_e_low = len_e - int((E_min_cut-func2(k_all[x],*popt))/(E_min_cut-E_max_cut)*len_e)
        pic[ind_e_up,x,:] = red
        pic[ind_e_low,x,:] = green
    plt.imshow(pic,cmap='gray')
    plt.show()
    exit()






