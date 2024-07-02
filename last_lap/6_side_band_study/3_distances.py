import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
s_ = 15
from scipy.optimize import curve_fit

sample = '3'
#sample = '11'

figname = "Figs/S"+sample+"_cuted.png"

im = Image.open(figname)

pic = np.array(np.asarray(im))

len_e, len_k, z = pic.shape

K_i = -0.6
K_f = 0.6
E_max_d = {'3':-0.5,'11':-0.9}
E_min_d = {'3':-1.7,'11':-2.2}
E_max = E_max_d[sample]
E_min = E_min_d[sample]

fig = plt.figure(figsize=(20,20))
#Cut at given E
plt.subplot(1,3,1)
y1_d = {'3':460, '11':395
        }
y1 = y1_d[sample]
E_cut = E_min+y1/len_e*(E_max-E_min)

#Offset
fit_pars_fn = "Data/S"+sample+"_fit_parameters.npy"
popt_fit = np.load(fit_pars_fn)
offset_E = popt_fit[1]
YYYY = (offset_E-E_min)/(E_max-E_min)*len_e

X = np.arange(len_k)
Y = 256-pic[len_e-y1,:,0]
def l3(x,w1,s1,p1,w2,s2,p2,w3,s3,p3):    #lorentz with three peaks
    return w1/((x-p1)**2+s1**2) +w2/((x-p2)**2+s2**2) + w3/((x-p3)**2+s3**2)

plt.title("Constant E="+"{:.2f}".format(E_cut-offset_E)+" eV cut (from VBM)")
x_d = {'3':383, '11':400}
x = x_d[sample]     #end_pixel in k direction for fit
centers_d = {'3':[149,227,288], '11':[155,243,336]}
c = centers_d[sample]
popt,pcov = curve_fit(l3,X[:x],Y[:x],p0=[80000,10,c[0],230000,10,c[1],150000,20,c[2]],
        bounds = (  (10 ,0  ,c[0]-20 ,100,0  ,c[1]-20,100,0  ,c[2]-20),
                    (1e8,50 ,c[0]+20 ,1e8,50 ,c[1]+20,1e8,80 ,c[2]+20)
            )
        )
print("Constant E plot:")
if 0:
    txt_par = ["w1","s1","p1","w2","s2","p2","w3","s3","p3"]
    for i in range(len(popt)):
        print(txt_par[i]+" ",popt[i])

print("Distance in k between main band and EXTERNAL side band: ","{:.4f}".format(abs(popt[2]-popt[5])/len_k*(K_f-K_i))+" A^{-1}")
plt.plot(X,Y,'b')
plt.plot(X[:x],l3(X[:x],*popt),'r',label='rw='+"{:.1f}".format(popt[0]/popt[3]*100)+"%")
plt.xticks([0,popt[2],popt[5],popt[8],len_k//2,len_k],["{:.2f}".format(K_i),"{:.2f}".format(K_i+popt[2]/len_k*(K_f-K_i)),"{:.2f}".format(K_i+popt[5]/len_k*(K_f-K_i)),"{:.2f}".format(K_i+popt[8]/len_k*(K_f-K_i)),"0","{:.2f}".format(K_f)])
for i in range(3):
    plt.plot([popt[2+i*3],popt[2+i*3]],[0,255],c='gray',lw=0.5,ls='dashed')
plt.xlabel(r"$\mathring{A}^{-1}$",size=s_)
plt.ylabel('intensity (0-255)',size=s_)
plt.ylim(0,255)
print("Relative weight: ",popt[0]/popt[3])
plt.legend(fontsize=s_)

#########################################################################
#########################################################################
#Cut at given k
plt.subplot(1,3,2)
y2 = int(popt[5])       #index of k where to take the cut
K_cut = K_i+y2/len_k*(K_f-K_i)
X = np.arange(len_e)
Y = 256-np.flip(pic[:,y2,0])

x_i_d = {'3':286, '11':240}
x_i = x_i_d[sample]
x_f = len_e
centers_d = {'3':[370,463,553], '11':[293,397,483]}
c = centers_d[sample]
popt2,pcov = curve_fit(l3,X[x_i:x_f],Y[x_i:x_f],p0=[570000,10,c[0],25832,10,c[1],300000,50,c[2]],
        bounds = (  (10 ,0  ,c[0]-20,10 ,0  ,c[1]-20,10 ,0  ,c[2]-20),
                    (1e8,200,c[0]+20,1e8,200,c[1]+20,1e8,200,c[2]+20)
            )
        )
print("\nConstant K plot:")
if 0:
    for i in range(len(popt2)):
        print(txt_par[i]+" ",popt2[i])
print("Distance in energy between main band and UP side band: ","{:.4f}".format(abs(popt2[5]-popt2[8])/len_e*(E_max-E_min)),' eV')

plt.plot(X,Y,'g')
plt.plot(X[x_i:x_f],l3(X[x_i:x_f],*popt2),'r',label='rw='+"{:.1f}".format(popt2[6]/popt2[3]*100)+"%")
plt.xticks([0,popt2[2],popt2[5],popt2[8],len_e],["{:.2f}".format(E_min),"{:.2f}".format(E_min+popt2[2]/len_e*(E_max-E_min)),"{:.2f}".format(E_min+popt2[5]/len_e*(E_max-E_min)),"{:.2f}".format(E_min+popt2[8]/len_e*(E_max-E_min)),"{:.2f}".format(E_max)])
for i in range(3):
    plt.plot([popt2[2+i*3],popt2[2+i*3]],[0,255],c='gray',lw=0.5,ls='dashed')
plt.xlabel("eV",size=s_)
plt.ylim(0,255)
print("Relative weight: ",popt2[6]/popt2[3])
plt.legend(fontsize=s_)

#########################################################################
#########################################################################
#########################################################################
#Figure meshgrid
plt.subplot(2,3,3)
pic2 = np.flip(pic,axis=0)

im = Image.fromarray(pic2) 
plt.imshow(im)
plt.hlines(y1,0,len_k,color='b')
plt.plot([y2,y2],[0,len_e],color='g')
plt.xticks([0,y2,len_k//2,len_k],["{:.2f}".format(K_i),"{:.2f}".format(K_cut),"{:.2f}".format((K_i+K_f)/2),"{:.2f}".format(K_f)])
plt.xlabel(r"$\mathring{A}^{-1}$",size=s_)
plt.yticks([0,y1,YYYY,len_e],["{:.2f}".format(E_min),"{:.2f}".format(E_cut),"{:.2f}".format(offset_E),"{:.2f}".format(E_max)])
plt.ylabel("eV",size=s_)

plt.hlines(YYYY,0,len_k,color='r',ls='dashed',lw=0.2)

plt.xlim(0,len_k)
plt.ylim(0,len_e)

if 0:
    #Intensities

    DE = (1.25-0.55)/len_e*abs(popt2[2]-popt2[-1])*1000
    DE2 = (1.25-0.55)/len_e*abs(popt2[2]-popt2[5])*1000
    list_V = np.linspace(1,50,101)
    rw = []
    for V in list_V:
        a = np.sqrt((DE/2/V)**2-1)      #dE/2/V
    #    rw_t = (a**2+1-a*np.sqrt(1+a**2))/(a**2+1+a*np.sqrt(1+a**2))
        rw_t = (a-np.sqrt(a**2+1))**(2)
        rw.append(rw_t)
    #print("V=",V)
    print("Distance in energy between the first two peaks: ",DE," meV")
    print("Distance in energy between the last two peaks: ",DE2," meV")
    #print("dE=",dE)
    #print("rel weight=",rw)
    plt.subplot(2,3,6)
    plt.plot(list_V,np.array(rw)*100,color='orange',label="relative weight(squared)")
    plt.plot(list_V,np.sqrt(np.array(rw))*100,color='pink',label="relative weight(lin)")
    mean_W = (W1+W2)/2
    plt.hlines(W1*100,list_V[0],list_V[-1],color='b')
    plt.hlines(W2*100,list_V[0],list_V[-1],color='g')
    plt.legend(fontsize=s_)
    plt.xlabel("V",size=s_)
    plt.ylabel("RW",size=s_)




fig.savefig('Figs/S'+sample+'_intensity_cuts.png')

plt.show()
