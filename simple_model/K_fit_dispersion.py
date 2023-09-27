import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

#Open image and take pixels
filename = "KK_LH_WSe2onWS2_forDario.png"
im  = Image.open(filename)

#to array
b_up = 12           ###Correct?
pic = np.asarray(im)[b_up:-b_up,b_up:-b_up]
pic = np.array(pic)
len_E,len_K,z = pic.shape
print(len_E,len_K)

new_image = Image.fromarray(np.uint8(pic))
new_imagename = "temp_1.png"
new_image.save(new_imagename)
#os.system("xdg-open "+new_imagename)

q = 200
m = 1.7
data = np.zeros((2,len_K-q),dtype=int)
#0.5 and 1.7 are different now..
xline = np.linspace(-1.1,-0.1,len_K)
yline = np.linspace(-1.4,-0.2,len_E)


#Take for each column the darkest point, above and below a separatrix line b/w the two bands, 
#and fit the points around it with a gaussian --> take max as darkest point
green = np.array([0,255,0,255])
red = np.array([255,0,0,255])
blue = np.array([0,0,255,255])
def inv_gauss(x,a,b,x0,s):
    return -(a*np.exp(-((x-x0)/s)**2)+b)
def find_max(col):
    #finds maximum of given column f pixels by fitting a gaussian
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
if 0:
    for i in range(q,len_K//2-q):
        e = int(len_E-i*m+q-1)
        pic[e,i] = blue
    new_image = Image.fromarray(np.uint8(pic))
    new_imagename = "temp_2.png"
    new_image.save(new_imagename)
    os.system("xdg-open "+new_imagename)
if 1:
    #new array
    for i in range(len_K//2-q):
        border = int(len_E-i*m+q) if i > q else len_E
        col = pic[:border,i,0]
        d = find_max(col)
        pic[d,i,:] = red
        data[0,i] = int(len_E-d)
#save new image
new_im = Image.fromarray(np.uint8(pic))
new_imagename = "extracted_pts.png"
new_im.save(new_imagename)
#os.system("xdg-open "+new_imagename)

#fitting of bands with simple model 
rem_d = y//10
comboY = np.append(yline[data[0,rem_d:-rem_d]],yline[data[1,rem_d:-rem_d]])
def func(k,m1,mu):
    return -k**2/2/m1 + mu

popt,pcov = curve_fit(
        func,
        comboX,comboY,
        p0=(-0.1,-1,0.7,0.1,0.1,-0.5),
        bounds=([-10,-10,0.1,0.01,0.01,-1],[10,10,2.7,5,5,-0.1]),
        )
print(popt)

#plot
plt.figure()
col = ['r','b']
for i in range(2):
    for j in range(0,y-2*y//10+1,10):
        plt.scatter(xline[j],yline[data[i,j+y//10]],color=col[i])
plt.plot(xline,func1(xline,*popt),'k-')
plt.plot(xline,func2(xline,*popt),'k-')
plt.xlim(-0.5,0.5)
plt.ylim(-1.7,-0.5)
plt.show()


#Save result of fitting
popt_filename = "popt_interlayer.npy"
np.save(popt_filename,popt)





