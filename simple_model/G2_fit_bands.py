import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

dirname = "figs_png/"
dirname_data = "data_fits/"
#Open image and take pixels
filename = dirname + "cut_KGK.png"
im  = Image.open(filename)

#to array
pic = np.array(np.asarray(im))
len_e,len_k,z = pic.shape

data = np.zeros((2,len_k),dtype=int)
r = 0.1         #removed k space from fit
E_min_cut = -1.7
E_max_cut = -0.5
K_cut = 0.5

k_line = np.linspace(-K_cut+r,K_cut-r,int(len_k*(1-r*2)))
e_line = np.linspace(E_min_cut,E_max_cut,len_e)

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
        border = len_e//2+(x-len_k//2)**2//400
        pic[border,x] = green
    new_image = Image.fromarray(np.uint8(pic))
    new_imagename = "temp.png"
    new_image.save(new_imagename)
    os.system("xdg-open "+new_imagename)
    exit()

new_filename = dirname + "G_extracted_points.png"
data_filename = dirname_data + "G_data_fit_bands.npy"
try:#compute darkest points of the two bands
    im  = Image.open(filename)
    data = np.load(data_filename)
except:
    for x in range(len_k):
        border = len_e//2+(x-len_k//2)**2//400
        col = pic[:border,x,0]
        d = find_max(col)
#        d = np.argmin(col)
        pic[d,x,:] = red
        data[0,x] = int(len_e-d)
    for x in range(len_k):
        border = len_e//2+(x-len_k//2)**2//400
        col = pic[border:,x,0]
        d = find_max(col)
#        d = np.argmin(col)
        pic[border+d,x,:] = blue
        data[1,x] = int(len_e-(border+d))
    #save data
    np.save(data_filename,data)
    #save new image
    new_im = Image.fromarray(np.uint8(pic))
    new_im.save(new_filename)

#fitting of bands with simple model 
comboX = np.append(k_line,k_line)
rem_d = int(len_k*r)+1           #?????
comboY = np.append(e_line[data[0,rem_d:-rem_d]],e_line[data[1,rem_d:-rem_d]])
def func1(k,a,b,c,m1,m2,mu):
    alpha = k**2/2/m1+k**2/2/m2+c
    beta = k**2/2/m1*(k**2/2/m2+c) - a**2*(1-b*k**2)**2
    step = np.sqrt(alpha**2-4*beta)
    return (-alpha+step)/2 + mu
def func2(k,a,b,c,m1,m2,mu):
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

popt,pcov = curve_fit(
        combinedFunc,
        comboX,comboY,
        p0=(-0.1,-1,0.7,0.1,0.1,-0.5),
        bounds=([-10,-10,0.1,0.01,0.01,-1],[10,10,2.7,5,5,-0.1]),
        )
print(popt)

if 1:#plot
    plt.figure()
    col = ['r','b']
    for i in range(2):
        for j in range(0,len_k-2*len_k//10+1,10):
            plt.scatter(k_line[j],e_line[data[i,j+len_k//10]],color=col[i])
    plt.plot(k_line,func1(k_line,*popt),'k-')
    plt.plot(k_line,func2(k_line,*popt),'k-')
    plt.xlim(-0.5,0.5)
    plt.ylim(-1.7,-0.5)
    plt.show()

#Save result of fitting
popt_filename = dirname_data + "G_popt_interlayer.npy"
np.save(popt_filename,popt)





