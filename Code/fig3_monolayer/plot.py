import sys,os
import numpy as np
cwd = os.getcwd()
master_folder = cwd[:40]
sys.path.insert(1, master_folder)
import CORE_functions as cfs
import functions as fs
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

save_E_int_fit = True
save_W_int_fit = True
save_orb = True
save_E_bands = True

""" Options plots """
s_norm = 10
s_small = 8
col_vline = 'b'
lw_vline = 0.5
ls_vline = (0, (10,7))
dict_box = dict(
    facecolor='white',   # box fill color
    edgecolor='black',   # border color
    linewidth=1,         # border thickness
    boxstyle='round',    # box shape
    alpha=0.8,           # transparency
    pad=0.3              # padding (in fraction of font size)
)
xl_text = 0.03
xr_text = 0.93
y_text = 0.86

plt.rcParams.update({
    "text.usetex": True,              # Use LaTeX for all text
    "font.family": "serif",           # Set font family
    "font.serif": ["Computer Modern"], # Default LaTeX font
    "text.latex.preamble": r"\usepackage{amsmath}",
})

fig = plt.figure( figsize=(7,4) )     #PRB is ~ 7x9.5 inches
gs = gridspec.GridSpec(
    2, 2,
    figure=fig,
    width_ratios=[2, 1]   # <-- first column twice as wide
)
gs_left = gridspec.GridSpecFromSubplotSpec(
    2, 1,
    subplot_spec=gs[:, 0],
    hspace=0.            # <-- custom vertical spacing
)

""" ARPES intensity """
TMD = 'WSe2'
ARPESfn = 'Inputs/intensity_GKM_%s_sum_BE_crop.txt'%TMD
ARPES_intensity = np.loadtxt(ARPESfn, delimiter="\t")
ARPES_intensity /= np.max(ARPES_intensity)

nK_int,nE = ARPES_intensity.shape
listE = np.linspace(-3.5,0,nE)
kEnd = 1.9896 if TMD=='WS2' else 1.9075
normK_ARPES = np.linspace(0,kEnd,nK_int,endpoint=True)

""" Intensity matrix of fit. """
# Evals and evecs
listK_fitI, norm_fitI = cfs.get_kList('G-K-M',nK_int,endpoint=True,returnNorm=True)
pars_en = ('intensity_fit_en',nK_int,TMD)
en_fn = cfs.getFilename(pars_en,dirname='Data/',extension='.npz')
if not Path(en_fn).is_file():
    fit_pars = np.load('Inputs/tb_%s.npy'%TMD)
    hopping = cfs.find_t(fit_pars)
    epsilon = cfs.find_e(fit_pars)
    offset = fit_pars[-3]
    HSO = cfs.find_HSO(fit_pars[-2:])
    args_H = (hopping,epsilon,HSO,cfs.dic_params_a_mono[TMD],offset)
    #
    all_H = cfs.H_monolayer(listK_fitI,*args_H)
    ens = np.zeros((nK_int,22))
    evs = np.zeros((nK_int,22,22),dtype=complex)
    for i in tqdm(range(nK_int),desc="Energies of fit intensity"):
        ens[i],evs[i] = np.linalg.eigh(all_H[i])
    if save_E_int_fit:
        np.savez(en_fn,ens=ens,evs=evs)
else:
    print("Loading energies of fit intensity")
    ens = np.load(en_fn)['ens']
    evs = np.load(en_fn)['evs']
# Intensity
typeSpread = 'Lorentz'    # 'Gauss' or 'Lorentz', works for both k and E
spreadK = 0.005     # in 1/a
spreadE = 0.05      # in eV
pars_spread = ('intensity_fit_we',nK_int,TMD,spreadK, spreadE, typeSpread)
we_fn = cfs.getFilename(pars_spread,dirname='Data/',extension='.npy')
if not Path(we_fn).is_file():
    weights = np.ones((nK_int,22))
    intensity_fit = np.zeros((nK_int,nE))
    for i in tqdm(range(nK_int),desc="Weights of fit intensity"):
        for n in range(22):
            intensity_fit += fs.weight_spreading(
                weights[i,n],
                listK_fitI[i],
                ens[i,n],
                listK_fitI,
                listE[None,:],
                pars_spread[-3:]
            )
    if save_W_int_fit:
        np.save(we_fn,intensity_fit)
else:
    print("Loading weights of fit intensity")
    intensity_fit = np.load(we_fn)
intensity_fit /= np.max(intensity_fit)

""" Orbital occupations DFT and fit """
nK_orb = 120
listK_orb, norm_orb = cfs.get_kList('G-K-M',nK_orb,endpoint=True,returnNorm=True)
pars_fn = ('orbitals',nK_orb,TMD)
orb_fn = cfs.getFilename(pars_fn,dirname='Data/',extension='.npz')
if not Path(orb_fn).is_file():
    fit_pars = np.load('Inputs/tb_%s.npy'%TMD)
    dft_pars = np.array(cfs.initial_pt[TMD])
    all_H_fit = cfs.H_monolayer(
        listK_orb,
        *(cfs.find_t(fit_pars),
          cfs.find_e(fit_pars),
          cfs.find_HSO(fit_pars[-2:]),
          cfs.dic_params_a_mono[TMD],
          fit_pars[-3]
          )
    )
    all_H_dft = cfs.H_monolayer(
        listK_orb,
        *(cfs.find_t(dft_pars),
          cfs.find_e(dft_pars),
          cfs.find_HSO(dft_pars[-2:]),
          cfs.dic_params_a_mono[TMD],
          dft_pars[-3]
          )
    )
    ens_fit = np.zeros((nK_orb,22))
    evs_fit = np.zeros((nK_orb,22,22),dtype=complex)
    ens_dft = np.zeros((nK_orb,22))
    evs_dft = np.zeros((nK_orb,22,22),dtype=complex)
    for i in tqdm(range(nK_orb),desc="Energies of orbitals"):
        ens_fit[i],evs_fit[i] = np.linalg.eigh(all_H_fit[i])
        ens_dft[i],evs_dft[i] = np.linalg.eigh(all_H_dft[i])
    """ Orbitals: d_xy, d_xz, d_z2, p_x, p_z """
    orbitals_fit = np.zeros((5,22,nK_orb))
    orbitals_dft = np.zeros((5,22,nK_orb))
    list_orbs = ([6,7],[0,1],[5,],[3,4,9,10],[2,8])
    for orb in range(5):
        inds_orb = list_orbs[orb]
        for ib in range(22):     #bands
            for ik in range(nK_orb):   #kpts
                for iorb in inds_orb:
                    orbitals_fit[orb,ib,ik] += np.linalg.norm(evs_fit[ik,iorb,ib])**2 + np.linalg.norm(evs_fit[ik,iorb+11,ib])**2
                    orbitals_dft[orb,ib,ik] += np.linalg.norm(evs_dft[ik,iorb,ib])**2 + np.linalg.norm(evs_dft[ik,iorb+11,ib])**2
    if save_orb:
        np.savez(orb_fn,
                 ens_fit=ens_fit,
                 orb_fit=orbitals_fit,
                 ens_dft=ens_dft,
                 orb_dft=orbitals_dft,
                 )
else:
    print("Loading orbitals fit and dft")
    ens_fit = np.load(orb_fn)['ens_fit']
    orbitals_fit = np.load(orb_fn)['orb_fit']
    ens_dft = np.load(orb_fn)['ens_dft']
    orbitals_dft = np.load(orb_fn)['orb_dft']

""" ARPES bands """
dataObject = cfs.dataWS2() if TMD=="WS2" else cfs.dataWSe2()
ptsPerPath = (30,15,10)
ARPES_bands = dataObject.getFitData(ptsPerPath)

""" Fit and DFT bands. """
args_e_data = ('bands_en',TMD,*ptsPerPath)
th_e_data_fn = cfs.getFilename(args_e_data,dirname='Data/',extension='.npz')
if not Path(th_e_data_fn).is_file():
    print("Computing energy bands")
    DFT_pars = np.array(cfs.initial_pt[TMD])  #DFT values of tb parameters. Order is: e, t, offset, SOC
    HSO_DFT = cfs.find_HSO(DFT_pars[-2:])
    bands_dft = cfs.energy(DFT_pars,HSO_DFT,ARPES_bands,TMD)
    #
    fit_pars = np.load("Inputs/tb_%s.npy"%TMD)
    HSO_fit = cfs.find_HSO(fit_pars[-2:])
    bands_fit = cfs.energy(fit_pars,HSO_fit,ARPES_bands,TMD)
    if save_E_bands:
        np.savez(th_e_data_fn,fit=bands_fit,dft=bands_dft)
else:
    print("Loading energy bands")
    bands_fit = np.load(th_e_data_fn)['fit']
    bands_dft = np.load(th_e_data_fn)['dft']

""" Plot intensity """
normK_full = np.concatenate([normK_ARPES,normK_ARPES[-1]+norm_fitI[-1]-norm_fitI[::-1]])
intensity_full = np.concatenate([ARPES_intensity**0.5,intensity_fit[::-1,:]],axis=0)
ax = fig.add_subplot(gs_left[0])
ax.pcolormesh(
    normK_full,
    listE,
    intensity_full.T,
    cmap='gray_r',
    rasterized=True
)
ax.set_ylabel("Energy [eV]",size=s_norm)
ax.tick_params(
    axis='y',
    which='both',
    left=True,
    right=True,
    labelleft=True,
    labelright=False,
    labelsize=s_norm
)
ax.set_xticks([])
for i in range(3):
    ind = [2,3,4]
    xval = normK_full[-1]/6*ind[i]
    if i==0:
        xval -= 1e-3
    ax.axvline(
        xval,
        color=col_vline,
        lw=lw_vline,
        ls=ls_vline,
        zorder=3
    )
ax.text(
    xl_text,
    y_text,
    "ARPES",
    transform=ax.transAxes,
    ha='left',
    va='center',
    fontsize=s_norm,
    bbox=dict_box,
)
ax.text(
    xr_text,
    y_text,
    "Fit",
    transform=ax.transAxes,
    ha='left',
    va='center',
    fontsize=s_norm,
    bbox=dict_box,
)

""" Plot orbitals """
ax = fig.add_subplot(gs_left[1])
color = ['red','brown','blue','green','aqua']
labels = [r"$d_{xy}+d_{x^2-y^2}$",r"$d_{xz}+d_{yz}$",r"$d_{z^2}$",r"$p_x+p_y$",r"$p_z$"]

leg1 = []
leg2 = []
for orb in range(5):
    for ib in range(22):
        ax.scatter(
            norm_orb,
            ens_dft[:,ib],
            s=(orbitals_dft[orb,ib]*30),
            marker='o',
            facecolor=color[orb],
            lw=0,
            alpha=0.3,
        )
        ax.scatter(
            2*norm_orb[-1]-norm_orb,
            ens_fit[:,ib],
            s=(orbitals_fit[orb,ib]*30),
            marker='o',
            facecolor=color[orb],
            lw=0,
            alpha=0.3,
        )

    if orb<3:
        leg1.append( Line2D([0], [0], marker='o',
                       markeredgecolor='none',
                       markerfacecolor=color[orb],
                       markersize=6,
                       label=labels[orb],
                       lw=0)
               )
    else:
        leg2.append( Line2D([0], [0], marker='o',
                       markeredgecolor='none',
                       markerfacecolor=color[orb],
                       markersize=6,
                       label=labels[orb],
                       lw=0)
               )
legend1 = ax.legend(handles=leg1,
                   loc=(0.23,0.33),
                   fontsize=s_small,
                   handletextpad=0.35,
                   handlelength=0.5,
                   labelspacing=0.1
                   )
legend2 = ax.legend(handles=leg2,
                   loc=(0.6,0.33),
                   fontsize=s_small,
                   handletextpad=0.35,
                   handlelength=0.5,
                   labelspacing=0.1
                   )
ax.add_artist(legend1)
ax.add_artist(legend2)
ax.set_ylim(listE[0],listE[-1])
ax.set_xlim(0,2*norm_orb[-1])

ax.set_xticks(
    [norm_orb[-1]/3*i for i in [0,2,3,4,6]],
    [r'$\Gamma$',r'$K$',r'$M$',r'$K$',r'$\Gamma$'],
    size=s_norm
)
ax.set_ylabel("Energy [eV]",size=s_norm)
ax.tick_params(
    axis='y',
    which='both',
    left=True,
    right=True,
    labelleft=True,
    labelright=False,
    labelsize=s_norm
)
for i in range(3):
    ind = [2,3,4]
    ax.axvline(
        norm_orb[-1]/3*ind[i],
        color=col_vline,
        lw=lw_vline,
        ls=ls_vline,
        zorder=3
    )

ax.text(
    xl_text,
    y_text,
    "DFT",
    transform=ax.transAxes,
    ha='left',
    va='center',
    fontsize=s_norm,
    bbox=dict_box,
)
ax.text(
    xr_text,
    y_text,
    "Fit",
    transform=ax.transAxes,
    ha='left',
    va='center',
    fontsize=s_norm,
    bbox=dict_box,
)

""" Plot bands """
ax = fig.add_subplot(gs[1,1])
lw = 0.7
for b in range(ARPES_bands.shape[1]-3):
    targ = np.argwhere(np.isfinite(ARPES_bands[:,3+b]))    #select only non-nan values
    xline = ARPES_bands[targ,0]
    xline = ARPES_bands[:,0]
    ax.plot(
        xline,
        ARPES_bands[:,3+b],
        color='r',
        marker='s',
        markersize=2,
        lw=0,
        label='ARPES' if b == 0 else '',
        zorder=-1,
        mec='k',
        mew=0.2
    )
    ax.plot(
        xline,
        bands_fit[b,:],
        color='blue',
        ls='-',
        lw=lw,
        label='Fit' if b == 0 else '',
        zorder=2,
        alpha=0.7
    )
    ax.plot(
        xline,
        bands_dft[b,:],
        color='orange',
        ls='-',
        lw=lw,
        label='DFT' if b == 0 else '',
        zorder=1,
        alpha=0.7
    )
leg = ax.legend(
    loc=(0.6,0.0),
    fontsize=s_small,
    labelspacing=0.1,
    facecolor='white',
    framealpha=1,
    edgecolor='black'
)
leg.get_frame().set_alpha(1)
ax.set_xlim(
    ARPES_bands[0,0],
    ARPES_bands[-1,0]
)
ax.set_ylim(
    listE[0],
    listE[-1]
)
ks = [ARPES_bands[0,0],4/3*np.pi/cfs.dic_params_a_mono[TMD],ARPES_bands[-1,0]]
ax.set_xticks(ks,[r"$\Gamma$",r"$K$",r"$M$"],size=s_norm)



plt.subplots_adjust(
    bottom = 0.064,
    top = 0.983,
    right = 0.974,
    left = 0.076,
    wspace = 0.136,
    hspace = 0.036
)

plt.show()

if 1:
    fig.savefig('Data/Fig_monolayer.pdf')
else:
    fig.savefig('Data/Fig_monolayer.png')


