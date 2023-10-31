import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

figname = "/home/dario/Desktop/git/MoireBands/0_simple_model/figs_png/cut_KGK_Moire.png"

im = Image.open(figname)

pic = np.array(np.asarray(im))

len_e, len_k, z = pic.shape

print(len_e,len_k)

plt.figure(figsize=(20,20))
#Cut at given E
plt.subplot(2,2,1)
y1 = 200
X = np.arange(len_k)
Y = 256-pic[len_e-y1,:,0]
def l2(x,w1,s1,p1,w2,s2,p2):#,w3,s3,p3):    #lorentz
    return w1/((x-p1)**2+s1**2) +w2/((x-p2)**2+s2**2) #+ w3/((x-p3)**2+s3**2)
def l3(x,w1,s1,p1,w2,s2,p2,w3,s3,p3):    #lorentz
    return w1/((x-p1)**2+s1**2) +w2/((x-p2)**2+s2**2) + w3/((x-p3)**2+s3**2)

def g2(x,w1,s1,p1,w2,s2,p2):#,w3,s3,p3):    #gauss
    return w1*np.exp(-((x-p1)/s1)**2) + w2*np.exp(-((x-p2)/s2)**2) #+ w3*np.exp(-((x-p3)/s3)**2)
def g3(x,w1,s1,p1,w2,s2,p2,w3,s3,p3):    #gauss
    return w1*np.exp(-((x-p1)/s1)**2) + w2*np.exp(-((x-p2)/s2)**2) + w3*np.exp(-((x-p3)/s3)**2)

if 0:       #gauss
    plt.title("Gauss")
    if 0:   #2 peaks
        print("gauss, 2 peaks")
        fun = g2
        x = 190
        popt,pcov = curve_fit(fun,X[:x],Y[:x],p0=[80,10,100,230,10,200],#,150,20,250],
                bounds = (  (10 ,0  ,50 ,100,0  ,140),#,100,0  ,180),
                            (120,100,150,250,100,200)#,200,80,270)
                    )
                )
    else:   #3 peaks
        print("gauss, 3 peaks")
        fun = g3
        x = 280
        popt,pcov = curve_fit(fun,X[:x],Y[:x],p0=[80,10,100,230,10,200,150,20,250],
                bounds = (  (10 ,0  ,50 ,100,0  ,140,100,0  ,180),
                            (120,100,150,250,100,200,200,80,270)
                    )
                )
else:   #lorentz
    plt.title("Lorentz")
    if 0:   #2 peaks
        print("lorentz, 2 peaks")
        fun = l2
        x = 190
        popt,pcov = curve_fit(fun,X[:x],Y[:x],p0=[80000,10,100,230000,10,200],#,150000,20,250],
                bounds = (  (10 ,0  ,50 ,100,0  ,140),#,100,0  ,180),
                            (1e8,50,150,1e8,50,200)#,1e8,80,270)
                    )
                )
    else:   #3 peaks
        print("lorentz, 3 peaks")
        fun = l3
        x = 280
        popt,pcov = curve_fit(fun,X[:x],Y[:x],p0=[80000,10,100,230000,10,200,150000,20,250],
                bounds = (  (10 ,0  ,50 ,100,0  ,140,100,0  ,180),
                            (1e8,50,150,1e8,50,200,1e8,80,270)
                    )
                )
txt_par = ["w1","s1","p1","w2","s2","p2","w3","s3","p3"]
print("Constant E plot:")
for i in range(len(popt)):
    print(txt_par[i]+" ",popt[i])

plt.plot(X,Y,'b')
plt.plot(X[:x],fun(X[:x],*popt),'r',label='rw='+"{:.1f}".format(popt[0]/popt[3]*100)+"%")
weights = np.array([popt[0],popt[3]])
W1 = (weights/np.max(weights))[0]
print("Relative weight: ",W1)
plt.legend()

#Cut at given k
plt.subplot(2,2,2)
y2 = int(popt[5])       #index of k where to take the cut
X = np.arange(len_e)
Y = 256-np.flip(pic[:,y2,0])

if 0:   #gauss
    plt.title("Gauss")
    x = 170
    fun = g2
    popt2,pcov = curve_fit(fun,X[x:],Y[x:],p0=[160,10,160,260,10,260],
            bounds = (  (50 ,0  ,100,50 ,0  ,200),
                        (500,100,250,500,100,350)
                )
            )
else:   #lorentz
    plt.title("Lorentz")
    if 0:   #2 peaks
        x = 170
        x_e = -1
        fun = l2
        popt2,pcov = curve_fit(fun,X[x:x_e],Y[x:x_e],p0=[150,10,193,80,10,288],
                bounds = (  (10 ,0  ,50 ,10 ,0  ,250),
                            (1e8,200,250,1e8,200,350)
                    )
                )
    else:   #3 peaks
        x = 50
        x_e = -1
        fun = l3
        popt2,pcov = curve_fit(fun,X[x:x_e],Y[x:x_e],p0=[570000,10,193,25832,10,288,300000,50,150],
                bounds = (  (10 ,0  ,50 ,10 ,0  ,250,10 ,0  ,x),
                            (1e8,200,250,1e8,200,350,1e8,200,180)
                    )
                )
print("\nConstant K plot:")
for i in range(len(popt2)):
    print(txt_par[i]+" ",popt2[i])

plt.plot(X,Y,'g')
plt.plot(X[x:],fun(X[x:],*popt2),'r',label='rw='+"{:.1f}".format(popt2[3]/popt2[0]*100)+"%")
weights = np.array([popt2[0],popt2[3]])
W2 = (weights/np.max(weights))[1]
print("Relative weight: ",W2)
plt.legend()

#Intensities

DE = (1.25-0.55)/len_e*abs(popt2[2]-popt2[5])*1000
list_V = np.linspace(1,50,101)
rw = []
for V in list_V:
    a = np.sqrt((DE/2/V)**2-1)      #dE/2/V
#    rw_t = (a**2+1-a*np.sqrt(1+a**2))/(a**2+1+a*np.sqrt(1+a**2))
    rw_t = (a-np.sqrt(a**2+1))**(2)
    rw.append(rw_t)
#print("V=",V)
print("Distance in energy between the two peaks: ",DE)
#print("dE=",dE)
#print("rel weight=",rw)
plt.subplot(2,2,3)
plt.plot(list_V,np.array(rw)*100,color='orange',label="relative weight(squared)")
plt.plot(list_V,np.sqrt(np.array(rw))*100,color='pink',label="relative weight(lin)")
mean_W = (W1+W2)/2
plt.hlines(W1*100,list_V[0],list_V[-1],color='b')
plt.hlines(W2*100,list_V[0],list_V[-1],color='g')
plt.legend()
plt.xlabel("V")

#Figure meshgrid
plt.subplot(2,2,4)
from matplotlib import cm
from matplotlib.colors import LogNorm
pic2 = np.flip(pic,axis=0)

im = Image.fromarray(pic2) 
plt.imshow(im)
plt.hlines(y1,0,len_k,color='b')
plt.vlines(y2,0,len_e,color='g')

plt.xlim(0,len_k)
plt.ylim(0,len_e)










plt.show()
