import numpy as np
import functions as fs
import sys,os

machine = fs.get_machine(os.getcwd())
"""
Here we compute the full image S11.
We need:
    - monolayer parameters
    - interlayer parameters
    - moire copies -> V and phi around G and K
"""

#Monolayer parameters
pars_mono = {}
hopping = {}
epsilon = {}
HSO = {}
offset = {}
for TMD in fs.materials:
    pars_mono[TMD] = np.load(fs.get_pars_mono_fn(TMD,machine))
    hopping[TMD] = fs.find_t(pars_mono[TMD])
    epsilon[TMD] = fs.find_e(pars_mono[TMD])
    HSO[TMD] = fs.find_HSO(pars_mono[TMD])
    offset[TMD] = pars_mono[TMD][-1]
#Interlayer parameters
pars_interlayer = np.load(fs.get_pars_interlayer_fn(machine))






