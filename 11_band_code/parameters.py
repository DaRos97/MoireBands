import numpy as np
#####
#####Parameters
#####
###Monolayer paerameters
#WS2 --> Table III  (first two in Angstrom, all others in eV, last in .. (lambda of SO)
dic_params_H = {    
        'WS2': {
            'e1':   ,
            'e3':   ,
            'e4':   ,
            'e6':   ,
            'e7':   ,
            'e9':   ,
            'e10':   ,
            't11':   ,
            't22':   ,
            't33':   ,
            't44':   ,
            't55':   ,
            't66':   ,
            't77':   ,
            't88':   ,
            't99':   ,
            't1010':   ,
            't1111':   ,
            't35':   ,
            't68':   ,
            't911':   ,
            't12':   ,
            't34':   ,
            't45':   ,
            't67':   ,
            't78':   ,
            't910':   ,
            't1011':   ,
            't41':   ,
            't32':   ,
            't52':   ,
            't96':   ,
            't116':   ,
            't107':   ,
            't98':   ,
            't118':   ,
            't96':   ,
            't116':   ,
            't98':   ,
            't118':   ,
                },
        'WSe2': {
            'e1':   ,
            'e3':   ,
            'e4':   ,
            'e6':   ,
            'e7':   ,
            'e9':   ,
            'e10':   ,
            't11':   ,
            't22':   ,
            't33':   ,
            't44':   ,
            't55':   ,
            't66':   ,
            't77':   ,
            't88':   ,
            't99':   ,
            't1010':   ,
            't1111':   ,
            't35':   ,
            't68':   ,
            't911':   ,
            't12':   ,
            't34':   ,
            't45':   ,
            't67':   ,
            't78':   ,
            't910':   ,
            't1011':   ,
            't41':   ,
            't32':   ,
            't52':   ,
            't96':   ,
            't116':   ,
            't107':   ,
            't98':   ,
            't118':   ,
            't96':   ,
            't116':   ,
            't98':   ,
            't118':   ,
                },
        }
dic_params_SO = {
        'W': 0.2874,
        'S': 0.0556.
        'Se': 0.2470,
        }
dic_params_a_mono = {
        'WS2': 3.18,
        'WSe2': 3.32,
        }
###Moirè potentials of bilayers. Different at Gamma (d_z^2 orbital) -> first two parameters, and K (d_xy orbitals) -> last two parameters
#WS2/WSe2 --> Gamma points from paper "G valley TMD moirè bands"(first in eV, second in radiants)
#WS2/WSe2 --> Louk's paper for K points(first in eV, second in radiants)
dic_params_V = {'WSe2/WS2':[0.0335,np.pi, 7.7*1e-3, -106*2*np.pi/360],
                'WS2/WSe2':[0.0335,np.pi, 7.7*1e-3, -106*2*np.pi/360],
            }
###Moirè length of bilayers in Angstrom
dic_a_M = { 'WS2/WSe2':79.8,
            'WSe2/WS2':79.8,
       }

