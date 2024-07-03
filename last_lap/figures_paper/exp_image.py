import numpy as np
import matplotlib.pyplot as plt
s_ = 20
from PIL import Image


S_fn = {'3':"../../Data_Experiments/S3_KGK_WSe2onWS2_v2.png",'11':"../../Data_Experiments/S11_KGK_WSe2onWS2_v1.png"}

WSe2 = 3.32
K_WSe2 = 4*np.pi/3/3.32

nf = 1
fig = plt.figure(figsize=(15,8))
for sample in ['3','11']:
    plt.subplot(1,2,nf)
    nf += 1
    im  = Image.open(S_fn[sample])
    original_pic = np.array(np.asarray(im))

    X_i = {'3':695-191, '11':808-204}
    X_f = {'3':original_pic.shape[1]-475+191, '11':original_pic.shape[1]-595+204}
    Y_i = {'3':108, '11':90}
    Y_f = {'3':original_pic.shape[0]-327, '11':original_pic.shape[0]-305}

    x_i = X_i[sample]
    y_i = Y_i[sample]
    x_f = X_f[sample]
    y_f = Y_f[sample]
    pic = np.array(np.asarray(im)[y_i:y_f,x_i:x_f])
    len_e,len_k,z = pic.shape
#    print((K_WSe2-1)/2*len_k)
#    continue
    E_max_fig = 0
    E_min_fig_d = {'3':-2.5, '11':-3.5}
    E_min_fig = E_min_fig_d[sample]
    K_fig = K_WSe2
    plt.imshow(pic,cmap='gray')
    plt.xticks([0,len_k//2,len_k],["{:.2f}".format(-K_fig),"0","{:.2f}".format(K_fig)])
    plt.yticks([0,len_e//2,len_e],["{:.2f}".format(E_max_fig),"{:.2f}".format((E_max_fig+E_min_fig)/2),"{:.2f}".format(E_min_fig)])
    plt.title("S"+sample,size=s_)
    plt.xlabel(r"$\mathring{A}^{-1}$",size=s_-5)
    plt.ylabel("eV",size=s_-5)

fig.savefig("Experimental_bands.png")
plt.show()
    
