import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import sys,os
cwd = os.getcwd()
master_folder = cwd[:43]
sys.path.insert(1, master_folder)
import CORE_functions as cfs
from pathlib import Path

plot_orb = True
plot_interlayer = True
plot_moire = True

color_bottom = 'dodgerblue'
color_top = 'orangered'

fig = plt.figure(figsize=(20,10))
"""
Fig 1.1 -> orbital content
We plot Gamma-K-M-Gamma two times, left for d orbitals and right for p orbitals.
"""
if plot_orb:
    #Parameters
    TMD = 'WSe2'
    Nmg = 100        #points between M and Gamma -> 100
    N2 = 5          #step  -> 3
    ens_fn = 'data/ens_'+TMD+'_'+str(Nmg)+'.npy'
    evs_fn = 'data/evs_'+TMD+'_'+str(Nmg)+'.npy'

    Ngk = int(Nmg*2/np.sqrt(3))
    Nkm = int(Nmg*1/np.sqrt(3))
    Nk = Ngk+Nkm+Nmg+1  #+1 so we compute G twice
    if Path(ens_fn).is_file() and Path(evs_fn).is_file():
        ens = np.load(ens_fn)
        evs = np.load(evs_fn)
    else:
        #Code
        a_TMD = cfs.dic_params_a_mono[TMD]
        par_values = np.array(cfs.initial_pt[TMD])  #DFT values
        #G-K-M-G
        K = np.array([4*np.pi/3/a_TMD,0])
        M = np.array([np.pi/a_TMD,np.pi/np.sqrt(3)/a_TMD])
        data = np.zeros((Nk,2))
        list_k = np.linspace(0,K[0],Ngk,endpoint=False)
        data[:Ngk,0] = list_k
        for ik in range(Nkm):
            data[Ngk+ik] = K + (M-K)/Nkm*ik
        for ik in range(Nmg):
            data[Ngk+Nkm+ik] = M - M/Nmg*ik
        #
        hopping = cfs.find_t(par_values)
        epsilon = cfs.find_e(par_values)
        offset = par_values[-3]
        #
        HSO = cfs.find_HSO(par_values[-2:])
        args_H = (hopping,epsilon,HSO,a_TMD,offset)
        #
        all_H = cfs.H_monolayer(data,*args_H)
        ens = np.zeros((Nk,22))
        evs = np.zeros((Nk,22,22),dtype=complex)
        for i in range(Nk):
            #index of TVB is 13, the other is 12 (out of 22: 11 bands times 2 for SOC. 7/11 are valence -> 14 is the TVB)
            ens[i],evs[i] = np.linalg.eigh(all_H[i])
        np.save(ens_fn,ens)
        np.save(evs_fn,evs)
    #Figure
    ax = fig.add_subplot(221)
    color = ['g','','m','pink','','r','b','','pink','m','']
    marker = ['s','','o','s','','o','^','','o','s','']
    xvals = np.linspace(0,Nk-1,Nk)
    for i in range(22):     #d-orbitals
        ax.plot(xvals,ens[:,i],'k-',lw=0.3,zorder=0)    #bands
        for orb in [5,6,0]:    #3 different d orbitals: [5->d_z2, 6-> d_xy, 0->d_xz]
            for ko in range(0,Nk,N2):   #kpts
                orb_content = np.linalg.norm(evs[ko,orb,i])**2 + np.linalg.norm(evs[ko,orb+11,i])**2    #add SOC
                if orb in [6,0]:    #add d_x2-y2 to d_xy and d_yz to d_xz
                    orb_content += np.linalg.norm(evs[ko,orb+1,i])**2 + np.linalg.norm(evs[ko,orb+1+11,i])**2
                ax.scatter(xvals[ko],ens[ko,i],s=orb_content*100,edgecolor=color[orb],marker=marker[orb],facecolor='none',lw=2,zorder=1)
    xvals = np.linspace(Nk,2*Nk-1,Nk)
    for i in range(22):     #same but for p-orbitals
        ax.plot(xvals,ens[:,i],'k-',lw=0.3,zorder=0)    #bands
        for orb in [2,3]:    #2 different p orbitals: 2->p_z_odd, 3->p_x_odd
            for ko in range(0,Nk,N2):   #kpts
                orb_content = np.linalg.norm(evs[ko,orb,i])**2 + np.linalg.norm(evs[ko,orb+11,i])**2        #SOC
                orb_content += np.linalg.norm(evs[ko,orb+6,i])**2 + np.linalg.norm(evs[ko,orb+6+11,i])**2   #add even
                if orb in [3,]: #add p_y to p_x
                    orb_content += np.linalg.norm(evs[ko,orb+1,i])**2 + np.linalg.norm(evs[ko,orb+1+11,i])**2
                    orb_content += np.linalg.norm(evs[ko,orb+6+1,i])**2 + np.linalg.norm(evs[ko,orb+1+6+11,i])**2
                ax.scatter(xvals[ko],ens[ko,i],s=orb_content*100,edgecolor=color[orb],marker=marker[orb],facecolor='none',lw=2,zorder=1)
    list_vline = [Ngk,Ngk+Nkm,Nk,Nk+Ngk,Nk+Ngk+Nkm] #vertical lines
    for i in list_vline:
        ax.axvline(i,lw=0.5,color='k',zorder=0)
    #Fermi level
    ax.axline(xy1=(0,0),slope=0,lw=0.5,color='k',zorder=0)
    #
    ax.set_xlim(0,2*Nk)
    #ax.set_ylim(mm,MM)
    ax.set_xticks([0,Ngk,Ngk+Nkm,Nk,Nk+Ngk,Nk+Ngk+Nkm,2*Nk],[r'$\Gamma$',r'$K$',r'$M$',r'$\Gamma$',r'$K$',r'$M$',r'$\Gamma$'])
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.set_ylabel("Energy (eV)",fontsize=20)
    #Legend
    names = [r'$d_{xz}+d_{yz}$','',r'$p_z$',r'$p_x+p_y$','',r'$d_{z^2}$',r'$d_{xy}+d_{x^2-y^2}$']
    leg1 = []
    for i in [5,6,0]:
        leg1.append( Line2D([0], [0], color="g", marker=marker[i], markeredgecolor=color[i], markeredgewidth=2, label=names[i],
                                  markerfacecolor='none', markersize=8, lw=0)
                   )
    legend1 = ax.legend(handles=leg1, loc='upper left',bbox_to_anchor=(0.32,0.74),fontsize=8,framealpha=1)
    ax.add_artist(legend1)
    leg2 = []
    for i in [2,3]:
        leg2.append( Line2D([0], [0], color="g", marker=marker[i], markeredgecolor=color[i], markeredgewidth=2, label=names[i],
                                  markerfacecolor='none', markersize=8, lw=0)
                   )
    legend2 = ax.legend(handles=leg2, loc='upper left',bbox_to_anchor=(0.52,0.715),fontsize=8,framealpha=1)
    ax.add_artist(legend2)
    #
"""
Fig. 1.2 -> interlayer stacking patterns and moirè unit cell
"""
if plot_interlayer:
    ax = fig.add_subplot(122)
    ax.axis('off')
    ax.set_aspect('equal')
    #Small Hexagons
    def plot_hex(center,radius,edgecolor,inverted,edges=['a','b','c','d','e','f']):
        angles = np.linspace(0+np.pi/6, 2 * np.pi+np.pi/6, 7)  # 6 sides + closing the loop
        x = center[0] + radius*np.cos(angles)
        y = center[1] + radius*np.sin(angles)
        ws = radius*0.3      #white space close to vertex
        lw = radius*2              #linewidth
        size_marker = radius*100
        list_vertices = []
        for e in edges:
            if e=='a':
                ax.plot([x[0]-ws/2*np.sqrt(3),x[1]+ws/2*np.sqrt(3)],[y[0]+ws/2,y[1]-ws/2],color=edgecolor,lw=lw)
                list_vertices += [0,1]
            if e=='d':
                ax.plot([x[4]-ws/2*np.sqrt(3),x[3]+ws/2*np.sqrt(3)],[y[4]+ws/2,y[3]-ws/2],color=edgecolor,lw=lw)
                list_vertices += [3,4]
            if e=='b':
                ax.plot([x[1]-ws/2*np.sqrt(3),x[2]+ws/2*np.sqrt(3)],[y[1]-ws/2,y[2]+ws/2],color=edgecolor,lw=lw)
                list_vertices += [1,2]
            if e=='e':
                ax.plot([x[5]-ws/2*np.sqrt(3),x[4]+ws/2*np.sqrt(3)],[y[5]-ws/2,y[4]+ws/2],color=edgecolor,lw=lw)
                list_vertices += [4,5]
            if e=='c':
                ax.plot([x[2],x[3]],[y[2]-ws,y[3]+ws],color=edgecolor,lw=lw)
                list_vertices += [2,3]
            if e=='f':
                ax.plot([x[0],x[5]],[y[0]-ws,y[5]+ws],color=edgecolor,lw=lw)
                list_vertices += [5,0]
        #Vertices
        list_vertices = list(set(list_vertices))
        markers = ['o', '+']  if inverted else ['+','o']
        for i in list_vertices:
            ax.scatter(x[i], y[i], color=edgecolor, marker=markers[i % 2], s=size_marker,facecolor='none' if markers[i%2]=='o' else edgecolor,lw=lw)
    radius = 1/np.sqrt(3)*1.2
    a1 = np.array([radius*np.sqrt(3),0])
    a2 = np.array([-radius*np.sqrt(3)/2,radius*3/2])
    shift_toplayer = np.array([0.1*radius,0*radius])
    columnx = 10
    shiftx1 = 2
    shiftx2 = 5.5
    shifty1 = 9
    shifty2 = 6
    shifty3 = 3.5
    shifty4 = 8     #big hex
    #MX_P
    center = np.array([columnx-shiftx1,shifty1])
    for x in range(2):
        for y in range(2):
            plot_hex(x*a1+y*a2+center,radius,color_bottom,False)
    shift_lattice = a1/3*2+a2/3
    plot_hex(shift_lattice+shift_toplayer+center,radius,color_top,False)
    plot_hex(a1+shift_lattice+shift_toplayer+center,radius,color_top,False,['d',])
    plot_hex(-a1+shift_lattice+shift_toplayer+center,radius,color_top,False,['e','b',])
    plot_hex(a2+shift_lattice+shift_toplayer+center,radius,color_top,False,['c','d','f',])
    #XM_P
    center = np.array([columnx-shiftx1,shifty2])
    for x in range(2):
        for y in range(2):
            plot_hex(x*a1+y*a2+center,radius,color_bottom,False)
    shift_lattice = np.array([0,radius])
    plot_hex(shift_lattice+shift_toplayer+center,radius,color_top,False)
    plot_hex(a1+shift_lattice+shift_toplayer+center,radius,color_top,False)
    plot_hex(-a2+shift_lattice+shift_toplayer+center,radius,color_top,False,['c','f'])
    #XX_P
    center = np.array([columnx-shiftx2,shifty3])
    plot_hex(center,radius,color_bottom,False)
    plot_hex(a1+center,radius,color_bottom,False)
    plot_hex(a1+a2+center,radius,color_bottom,False)
    plot_hex(shift_toplayer+center,radius,color_top,False)
    plot_hex(a1+shift_toplayer+center,radius,color_top,False)
    plot_hex(a1+a2+shift_toplayer+center,radius,color_top,False,)
    #XX_AP
    center = np.array([columnx+shiftx1,shifty1])
    for x in range(2):
        for y in range(2):
            plot_hex(x*a1+y*a2+center,radius,color_bottom,False)
    shift_lattice = np.array([0,radius])
    plot_hex(shift_lattice+shift_toplayer+center,radius,color_top,True)
    plot_hex(a1+shift_lattice+shift_toplayer+center,radius,color_top,True)
    plot_hex(-a2+shift_lattice+shift_toplayer+center,radius,color_top,True,['c','f'])
    #2H_AP
    center = np.array([columnx+shiftx1,shifty2])
    plot_hex(center,radius,color_bottom,False)
    plot_hex(a1+center,radius,color_bottom,False)
    plot_hex(a1+a2+center,radius,color_bottom,False)
    plot_hex(shift_toplayer+center,radius,color_top,True)
    plot_hex(a1+shift_toplayer+center,radius,color_top,True)
    plot_hex(a1+a2+shift_toplayer+center,radius,color_top,True,)
    #MM_AP
    center = np.array([columnx+shiftx2,shifty3])
    for x in range(2):
        for y in range(2):
            plot_hex(x*a1+y*a2+center,radius,color_bottom,False)
    shift_lattice = a1/3*2+a2/3
    plot_hex(shift_lattice+shift_toplayer+center,radius,color_top,True)
    plot_hex(a1+shift_lattice+shift_toplayer+center,radius,color_top,True,['d',])
    plot_hex(-a1+shift_lattice+shift_toplayer+center,radius,color_top,True,['e','b',])
    plot_hex(a2+shift_lattice+shift_toplayer+center,radius,color_top,True,['c','d','f',])
    #Big Hexagons
    big_radius = 3*radius
    center1 = (columnx-shiftx2,shifty4)
    center2 = (columnx+shiftx2+.5,shifty4)
    big_lw = 2*radius
    he1 = patches.RegularPolygon(center1, numVertices=6, radius=big_radius, orientation=0, edgecolor='k', facecolor='none', zorder=0, lw=big_lw)
    he2 = patches.RegularPolygon(center2, numVertices=6, radius=big_radius, orientation=0, edgecolor='k', facecolor='none', zorder=0, lw=big_lw)
    ax.add_patch(he1)
    ax.add_patch(he2)
    arc_radius = big_radius*0.4
    ls = (0,(3,5,1,5))
    circ1a = patches.Arc((center1[0]+big_radius/2*np.sqrt(3),center1[1]+big_radius/2),arc_radius,arc_radius,theta1=150,theta2=270,ls=ls,color='k',lw=big_lw)
    circ1b = patches.Arc((center1[0]+big_radius/2*np.sqrt(3),center1[1]-big_radius/2),arc_radius,arc_radius,theta1=90,theta2=210,ls=ls,color='k',lw=big_lw)
    circ1c = patches.Arc(center1,arc_radius,arc_radius,theta1=0,theta2=360,ls=ls,color='k',lw=big_lw)
    ax.add_patch(circ1a)
    ax.add_patch(circ1b)
    ax.add_patch(circ1c)
    circ2a = patches.Arc((center2[0]-big_radius/2*np.sqrt(3),center2[1]+big_radius/2),arc_radius,arc_radius,theta1=270,theta2=30,ls=ls,color='k',lw=big_lw)
    circ2b = patches.Arc((center2[0]-big_radius/2*np.sqrt(3),center2[1]-big_radius/2),arc_radius,arc_radius,theta1=330,theta2=90,ls=ls,color='k',lw=big_lw)
    circ2c = patches.Arc(center2,arc_radius,arc_radius,theta1=0,theta2=360,ls=ls,color='k',lw=big_lw)
    ax.add_patch(circ2a)
    ax.add_patch(circ2b)
    ax.add_patch(circ2c)
    #Separation and title
    ytitle = shifty1+2
    ax.plot([columnx+0.5,columnx+0.5],[-0.5,ytitle],color='k',lw=2,ls='dashed')
    xtitle = shiftx2
    fs_title = 20
    ax.text(columnx-xtitle-1,ytitle,'Parallel',fontsize=fs_title)
    ax.text(columnx+xtitle-1,ytitle,'Anti-Parallel',fontsize=fs_title)
    #Legend
    fs_legend = 18
    leg = []
    ms = 10
    leg.append( Line2D([0], [0], color='k', marker='o', markeredgecolor=color_top, markeredgewidth=2, label=r'$M^{\text{top}}$',
                                  markerfacecolor='none', markersize=ms, lw=0)       )
    leg.append( Line2D([0], [0], color='k', marker='+', markeredgecolor=color_top, markeredgewidth=2, label=r'$X_2^{\text{top}}$',
                                  markerfacecolor='none', markersize=ms, lw=0)       )
    leg.append( Line2D([0], [0], color='k', marker='o', markeredgecolor=color_bottom, markeredgewidth=2, label=r'$M^{\text{bottom}}$',
                                  markerfacecolor='none', markersize=ms, lw=0)       )
    leg.append( Line2D([0], [0], color='k', marker='+', markeredgecolor=color_bottom, markeredgewidth=2, label=r'$X_2^{\text{bottom}}$',
                                  markerfacecolor='none', markersize=ms, lw=0)       )
    yleg = 0.4
    legend = ax.legend(handles=leg, loc='upper left',bbox_to_anchor=(0.31,yleg),fontsize=fs_legend,framealpha=1,ncol=2)
    ax.add_artist(legend)
    #Equations
    yeq = 0.5
    xeq = 2
    fs_eq = 25
    ax.text(xeq+0.5,yeq,r'$t_0^{\text{(P)}}({\bf k})=-a-b\sum_{j=1}^6e^{-i{\bf k}\cdot{\bf e}_j}$',fontsize=fs_eq)
    ax.text(columnx+xeq,yeq,r'$t_0^{\text{(AP)}}({\bf k})=-b\sum_{j=1}^3e^{-i{\bf k}\cdot{\bf \delta}_j}$',fontsize=fs_eq)

"""
Fig. 1.3 -> moire period in real and momentum space and scheme/numbering of mini-BZs
"""
if plot_moire:
    ax = fig.add_subplot(2,2,3)
    ax.axis('off')
#    ax.set_xlim(-2,15)
#    ax.set_ylim(-3,3)
    ax.set_aspect('equal')
    radiuses = [1.3,1.18]
    colors = [color_top,color_bottom]
    hw = 0.15
    hl = 0.21
    fs_vec = 15
    #Real space
    center_r = np.array([0,0])
    lw = 2
    for i in range(1,-1,-1):
        radius = radiuses[i]
        a1 = np.array([radius*np.sqrt(3),0])
        a2 = np.array([-radius*np.sqrt(3)/2,radius*3/2])
        hes = []
        hes.append( patches.RegularPolygon(center_r, numVertices=6, radius=radius, orientation=0, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
        hes.append( patches.RegularPolygon(center_r+a1, numVertices=6, radius=radius, orientation=0, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
        hes.append( patches.RegularPolygon(center_r+a1+a2, numVertices=6, radius=radius, orientation=0, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
        hes.append( patches.RegularPolygon(center_r+a2, numVertices=6, radius=radius, orientation=0, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
        hes.append( patches.RegularPolygon(center_r-a1, numVertices=6, radius=radius, orientation=0, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
        hes.append( patches.RegularPolygon(center_r-a1-a2, numVertices=6, radius=radius, orientation=0, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
        hes.append( patches.RegularPolygon(center_r-a2, numVertices=6, radius=radius, orientation=0, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
        for j in range(len(hes)):
            ax.add_patch(hes[j])
        if i==0:
            ax.arrow(center_r[0],center_r[1],a1[0],0,head_width=hw,head_length=hl,length_includes_head=True,facecolor='k')
            ax.arrow(center_r[0],center_r[1],a2[0],a2[1],head_width=hw,head_length=hl,length_includes_head=True,facecolor='k')
            ax.text(center_r[0]+a1[0],center_r[1]+radius/4,r'${\bf a}_1$',fontsize=fs_vec)
            ax.text(center_r[0]+a2[0],center_r[1]+a2[1]+radius/4,r'${\bf a}_2$',fontsize=fs_vec)
    #Momentum space
    center_k = np.array([5.5*radiuses[0],0])
    for i in range(2):
        radius = radiuses[(i+1)%2]
        b1 = np.array([radius*3/2,radius*np.sqrt(3)/2])
        b2 = np.array([0,radius*np.sqrt(3)])
        ang = np.pi/6
        hes = []
        hes.append( patches.RegularPolygon(center_k, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
        hes.append( patches.RegularPolygon(center_k+b1, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
        hes.append( patches.RegularPolygon(center_k+b2, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
        hes.append( patches.RegularPolygon(center_k-b1+b2, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
        hes.append( patches.RegularPolygon(center_k-b1, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
        hes.append( patches.RegularPolygon(center_k+b1-b2, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
        hes.append( patches.RegularPolygon(center_k-b2, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
        for j in range(len(hes)):
            ax.add_patch(hes[j])
        if i==0:
            ax.arrow(center_k[0],center_k[1],b1[0],b1[1],head_width=hw,head_length=hl,length_includes_head=True,facecolor='k')
            ax.arrow(center_k[0],center_k[1],b2[0],b2[1],head_width=hw,head_length=hl,length_includes_head=True,facecolor='k')
            ax.text(center_k[0]+b1[0],center_k[1]+b1[1]+radius/4,r'${\bf b}_1$',fontsize=fs_vec)
            ax.text(center_k[0]+b2[0],center_k[1]+b2[1]+radius/4,r'${\bf b}_2$',fontsize=fs_vec)
    #mini-BZ
    center_m = np.array([10*radiuses[0],0])
    radius = 0.4
    g1 = np.array([radius*3/2,radius*np.sqrt(3)/2])
    g2 = np.array([0,radius*np.sqrt(3)])
    ang = np.pi/6
    colors = ['coral',]
    i = 0
    lw = 1
    hes = []
    hes.append( patches.RegularPolygon(center_m, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
    hes.append( patches.RegularPolygon(center_m+g1, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
    hes.append( patches.RegularPolygon(center_m+g2, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
    hes.append( patches.RegularPolygon(center_m-g1+g2, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
    hes.append( patches.RegularPolygon(center_m-g1, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
    hes.append( patches.RegularPolygon(center_m+g1-g2, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
    hes.append( patches.RegularPolygon(center_m-g2, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
    hes.append( patches.RegularPolygon(center_m+2*g1, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
    hes.append( patches.RegularPolygon(center_m+g1+g2, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
    hes.append( patches.RegularPolygon(center_m+2*g2, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
    hes.append( patches.RegularPolygon(center_m-g1+2*g2, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
    hes.append( patches.RegularPolygon(center_m-2*g1+2*g2, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
    hes.append( patches.RegularPolygon(center_m-2*g1+g2, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
    hes.append( patches.RegularPolygon(center_m-2*g1, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
    hes.append( patches.RegularPolygon(center_m-g1-g2, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
    hes.append( patches.RegularPolygon(center_m-2*g2, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
    hes.append( patches.RegularPolygon(center_m+g1-2*g2, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
    hes.append( patches.RegularPolygon(center_m+2*g1-2*g2, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
    hes.append( patches.RegularPolygon(center_m+2*g1-g2, numVertices=6, radius=radius, orientation=ang, edgecolor=colors[i], facecolor='none', zorder=0, lw=lw))
    for j in range(len(hes)):
        ax.add_patch(hes[j])
    ax.arrow(center_m[0],center_m[1],g1[0],g1[1],head_width=hw,head_length=hl,length_includes_head=True,facecolor='k')
    ax.arrow(center_m[0],center_m[1],g2[0],g2[1],head_width=hw,head_length=hl,length_includes_head=True,facecolor='k')
    ax.text(center_m[0]+g1[0],center_m[1]+g1[1]+radius/4,r'${\bf G}_1$',fontsize=fs_vec)
    ax.text(center_m[0]+g2[0],center_m[1]+g2[1]+radius/4,r'${\bf G}_2$',fontsize=fs_vec)
    #Titles
    fs_title = 20
    ytit = 3*radiuses[0]
    ax.text(center_r[0]-1.5,ytit,'Real space',fontsize=fs_title)
    ax.text(center_k[0]-2.2,ytit,'Momentum space',fontsize=fs_title)
    ax.text(center_m[0]-2,ytit-1.5,'mini-Brillouin zones',fontsize=fs_title)
    #Ellipse and arrow between k and mini
    col = 'cyan'
    ax.add_patch(patches.Arc((center_k[0]+(radiuses[0]+radiuses[1])/2,0),width=radiuses[0]/2,height=radiuses[0]/3,color=col ))
    ax.add_patch(patches.Arc((center_m[0],center_m[1]-radius/2*np.sqrt(3)),width=radius*2,height=radius,color=col ))
    ax.arrow(center_k[0]+(radiuses[0]+radiuses[1])/2+radiuses[0]/4,                             #x
             0,                                                                                 #y
             center_m[0]-radius-center_k[0]-(radiuses[0]+radiuses[1])/2-radiuses[0]/4,        #dx
             -radius/2*np.sqrt(3),                                                              #dy
             head_width=0.3,head_length=0.5,length_includes_head=True,color=col)


























































#   
fig.tight_layout()
#plt.subplots_adjust(wspace=0.239)
plt.show()

