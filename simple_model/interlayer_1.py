import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

filename = "KGK_WSe2onWS2_forDario.png"
im  = Image.open(filename)

#to array
b_up = 12
pic = np.asarray(im)[b_up:-b_up,b_up:-b_up]
pic = np.array(pic)
x,y,z = pic.shape

data = np.zeros((2,y),dtype=int)
r = 0.1
xline = np.linspace(-0.5+r,0.5-r,y-2*y//10+1)
yline = np.linspace(-1.7,-0.5,x)

green = np.array([0,255,0,255])
red = np.array([255,0,0,255])
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
if 1:
    #new array
    for i in range(y):
        border = x//2+(i-y//2)**2//700
        col = pic[:border,i,0]
        d = find_max(col)
#        d = np.argmin(col)
        pic[d,i,:] = red
        data[0,i] = int(x-d)
    for i in range(y):
        border = x//2+(i-y//2)**2//700
        col = pic[border:,i,0]
        d = find_max(col)
#        d = np.argmin(col)
        pic[border+d,i,:] = blue
        data[1,i] = int(x-(border+d))

#save new image
new_im = Image.fromarray(np.uint8(pic))
new_filename = "test_im.png"
new_im.save(new_filename)

#fitting
comboX = np.append(xline,xline)
rem_d = y//10
comboY = np.append(yline[data[0,rem_d:-rem_d]],yline[data[1,rem_d:-rem_d]])
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
    extract1 = combK[:len(xline)]
    extract2 = combK[len(xline):]

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



popt_filename = "popt_interlayer.npy"
np.save(popt_filename,popt)





