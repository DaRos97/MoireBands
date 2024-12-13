import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
s_ = 15
from scipy.optimize import curve_fit
import sys

#sample = '3'
#sample = '11'
sample = sys.argv[1]
print("Computing side band distance of sample "+sample)

dirname = "Figs/"
dirname_data = "Data/"
#Open cuted image
image_fn = dirname_data + "S"+sample+"_cuted.npy"
pic = np.load(image_fn)
len_e, len_k, z = pic.shape
#Offset
fit_pars_fn = dirname_data + "S"+sample+"_fit_parameters.npy"
popt_fit = np.load(fit_pars_fn)

K_i = -0.6
K_f = 0.6
E_max_d = {'3':-0.5,'11':-0.9}
E_min_d = {'3':-1.7,'11':-2.1}
E_max = E_max_d[sample]
E_min = E_min_d[sample]

if 0:   #Plot cuted image with parabula and offset to make sure we are using same units as for the fit
    fig = plt.figure()
    plt.imshow(pic)
    new_k = np.arange(len_k)
    def func(k,m,offset):
        return -k**2/2/m + offset
    new_parabola = len_e*(E_max-func(np.linspace(K_i,K_f,len_k),*popt_fit))/(E_max-E_min)
    plt.plot(new_k,new_parabola,'g')
    y_off = len_e-abs((E_min-popt_fit[1])/(E_min-E_max)*len_e)
    plt.plot([0,len_k],[y_off,y_off],'b')
    #
    plt.xlim(0,len_k)
    plt.ylim(len_e,0)
    plt.xticks([0,len_k//2,len_k],["{:.1f}".format(K_i),"0","{:.1f}".format(K_f)])
    plt.yticks([0,len_e//2,len_e],["{:.1f}".format(E_max),"{:.1f}".format((E_max+E_min)/2),"{:.1f}".format(E_min)])
    plt.show()
    exit()

fig = plt.figure(figsize=(20,6))
#Cut at given E
plt.subplot(1,3,1)
offset_E = popt_fit[1]
ind_offsett = int(len_e*(E_max-offset_E)/(E_max-E_min))
result_E_cut_VBM = 0.28      #This parameters decides essentially everything, is the energy distance from the VBM

E_cut = offset_E - result_E_cut_VBM
ind_E_cut = int(len_e*(E_max-E_cut)/(E_max-E_min))

if 0:   #plot energy cut
    print(E_cut)
    print(ind_E_cut)
    fig = plt.figure()
    plt.imshow(pic)
    plt.hlines(ind_E_cut,0,len_k,color='b')
    plt.show()
    exit()

X = np.arange(len_k)
Y = 256-pic[ind_E_cut,:,0]
def l3(x,w1,s1,p1,w2,s2,p2,w3,s3,p3):    #lorentz with three peaks
    return w1/((x-p1)**2+s1**2) +w2/((x-p2)**2+s2**2) + w3/((x-p3)**2+s3**2)

plt.title("Constant E="+"{:.2f}".format(result_E_cut_VBM)+" eV cut (from VBM)")
x_d = {'3':383, '11':400}
x_end = x_d[sample]     #end_pixel in k direction for fit
centers_d = {'3':[149,227,295], '11':[140,240,335]}
rg_peak = int(0.04/(K_f-K_i)*len_k)     #Give a range for finding the peak of 0.05 A^-1
c = centers_d[sample]
popt,pcov = curve_fit(l3,X[:x_end],Y[:x_end],p0=[1e6,10,c[0],1e6,10,c[1],1e6,20,c[2]],
        bounds = (  (1e2,0  ,c[0]-rg_peak ,1e2,0  ,c[1]-rg_peak,1e2,0  ,c[2]-rg_peak),
                    (1e8,50 ,c[0]+rg_peak ,1e8,50 ,c[1]+rg_peak,1e8,80 ,c[2]+rg_peak)
            )
        )
print("Constant E plot:")
txt_par = ["w1","s1","p1","w2","s2","p2","w3","s3","p3"]
if 1:
    for i in range(len(popt)):
        print(txt_par[i]+" ",popt[i])
    print("Relative weight using Lorentz parameters: ",popt[0]/popt[3])
    print("\nIntensity at left peak:",256-pic[ind_E_cut,int(popt[2]),0])
    print("\nIntensity at central peak:",256-pic[ind_E_cut,int(popt[5]),0])
    print("Relative weight using intensity peaks: ",(256-pic[ind_E_cut,int(popt[2]),0])/(256-pic[ind_E_cut,int(popt[5]),0]))

result_dist_k = abs(popt[2]-popt[5])/len_k*(K_f-K_i)
print("Distance in k between main band and EXTERNAL side band: ","{:.4f}".format(result_dist_k)+" A^{-1}")
plt.plot(X,Y,'b')
plt.plot(X[:x_end],l3(X[:x_end],*popt),'r')
plt.xticks([0,popt[2],popt[5],popt[8],len_k//2,len_k],["{:.2f}".format(K_i),"{:.2f}".format(K_i+popt[2]/len_k*(K_f-K_i)),"{:.2f}".format(K_i+popt[5]/len_k*(K_f-K_i)),"{:.2f}".format(K_i+popt[8]/len_k*(K_f-K_i)),"0","{:.2f}".format(K_f)])
for i in range(3):
    plt.plot([popt[2+i*3],popt[2+i*3]],[0,255],c='gray',lw=0.5,ls='dashed')
plt.xlabel(r"$\mathring{A}^{-1}$",size=s_)
plt.ylabel('intensity (0-255)',size=s_)
plt.ylim(0,255)
#plt.legend(fontsize=s_)

#########################################################################
#########################################################################
#Cut at given k
plt.subplot(1,3,2)
y2 = int(popt[5])       #index of k where to take the cut
K_cut = K_i+y2/len_k*(K_f-K_i)
result_K_cut = K_cut
X = np.arange(len_e)
Y = 256-np.flip(pic[:,y2,0])

x_i_d = {'3':286, '11':200}
x_i = x_i_d[sample]
x_f = len_e-90 if sample=='11' else len_e
centers_d = {'3':[370,463,560], '11':[220,340,430]}
c = centers_d[sample]
rg_peak = int(0.08/(E_max-E_min)*len_e)     #Give a range for finding the peak of 0.08 eV
popt2,pcov = curve_fit(l3,X[x_i:x_f],Y[x_i:x_f],p0=[4e6,70,c[0],4e6,47,c[1],2e3,15,c[2]],
        bounds = (  (1e2,0  ,c[0]-rg_peak,1e2,0  ,c[1]-rg_peak,1e2,0  ,c[2]-10),
                    (1e8,200,c[0]+rg_peak,1e8,200,c[1]+rg_peak,1e8,30,c[2]+10)
            )
        )
print("\nConstant K plot:")
if 1:
    for i in range(len(popt2)):
        print(txt_par[i]+" ",popt2[i])
result_dist_E = abs(popt2[5]-popt2[8])/len_e*(E_max-E_min)
print("Distance in energy between main band and UP side band: ","{:.4f}".format(result_dist_E),' eV')

plt.title("Constant K="+"{:.2f}".format(K_cut)+r" $\mathring{A}^{-1}$ cut")
plt.plot(X,Y,'g')
#popt2[-3:] = [2e3,15,430]
plt.plot(X[x_i:x_f],l3(X[x_i:x_f],*popt2),'r')
#plt.xticks([0,popt2[2],popt2[5],popt2[8],len_e],["{:.2f}".format(E_min),"{:.2f}".format(E_min+popt2[2]/len_e*(E_max-E_min)),"{:.2f}".format(E_min+popt2[5]/len_e*(E_max-E_min)),"{:.2f}".format(E_min+popt2[8]/len_e*(E_max-E_min)),"{:.2f}".format(E_max)])
for i in range(3):
    plt.plot([popt2[2+i*3],popt2[2+i*3]],[0,255],c='gray',lw=0.5,ls='dashed')
plt.xlabel("eV",size=s_)
plt.ylim(0,255)
#plt.legend(fontsize=s_)

#########################################################################
#########################################################################
#########################################################################
#Figure
plt.subplot(1,3,3)
plt.imshow(pic)

plt.hlines(ind_E_cut,0,len_k,color='b')
plt.plot([y2,y2],[0,len_e],color='g')
plt.xticks([0,y2,len_k//2,len_k],["{:.2f}".format(K_i),"{:.2f}".format(K_cut),"{:.2f}".format((K_i+K_f)/2),"{:.2f}".format(K_f)])
plt.xlabel(r"$\mathring{A}^{-1}$",size=s_)
plt.yticks([len_e,ind_E_cut,0],["{:.2f}".format(E_min),"{:.2f}".format(E_cut),"{:.2f}".format(E_max)])
plt.ylabel("eV",size=s_)

plt.xlim(0,len_k)
plt.ylim(len_e,0)

if 1:
    fig.savefig('Figs/S'+sample+'_intensity_cuts.png')

plt.show()
