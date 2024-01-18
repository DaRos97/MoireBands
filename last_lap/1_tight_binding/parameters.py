import numpy as np

#BZ cuts of exp data
paths = ['KGK','KMKp']

###Monolayer DFT paerameters. 11-band tight binding model. 
# --> TO CHECK
#WS2 --> Table III  (first two in Angstrom, all others in eV, last in .. (lambda of SO)
initial_pt = {    
        'WS2': [
            #'e1':   
            1.3754,
            #'e3':   
            -1.1278,
            #'e4':   
            -1.5534,
            #'e6':   
            -0.0393,
            #'e7':   
            0.1984,
            #'e9':   
            -3.3706,
            #'e10':  
            -2.3461,

            #'t1_11':   
            -0.2011,
            #'t1_22':   
            0.0263,
            #'t1_33':   
            -0.1749,
            #'t1_44':   
            0.8726,
            #'t1_55':   
            -0.2187,
            #'t1_66':   
            -0.3716,
            #'t1_77':   
            0.3537,
            #'t1_88':   
            -0.6892,
            #'t1_99':   
            -0.2112,
            #'t1_1010':   
            0.9673,
            #'t1_1111':     17 
            0.0143,
            #'t1_35':   
            -0.0818,
            #'t1_68':   
            0.4896,
            #'t1_911':   
            -0.0315,
            #'t1_12':   
            -0.3106,
            #'t1_34':   
            -0.1105,
            #'t1_45':   
            -0.0989,
            #'t1_67':   
            -0.1467,
            #'t1_78':   
            -0.3030,
            #'t1_910':   
            0.1645,
            #'t1_1011':   
            -0.1018,

            #'t5_41':   
            -0.8855,
            #'t5_32':   
            -1.4376,
            #'t5_52':   
            2.3121,
            #'t5_96':   
            -1.0130,
            #'t5_116':   
            -0.9878,
            #'t5_107':   
            1.5629,
            #'t5_98':   
            -0.9491,
            #'t5_118':   
            0.6718,

            #'t6_96':   
            -0.0659,
            #'t6_116':   
            -0.1533,
            #'t6_98':   
            -0.2618,
            #'t6_118':          39 
            -0.2736,

            #SO
            #'W':
            0.2874,
            #'S'
            0.0556,

            #'offset
            -1.34,

                ],
        'WSe2': [
            #'e1':   
            1.0349,
            #'e3':   
            -0.9573,
            #'e4':   
            -1.3937,
            #'e6':   
            -0.1667,
            #'e7':   
            0.0984,
            #'e9':   
            -3.3642,
            #'e10':   
            -2.1820,

            #'t1_11':   
            -0.1395,
            #'t1_22':   
            0.0129,
            #'t1_33':   
            -0.2171,
            #'t1_44':   
            0.9763,
            #'t1_55':   
            -0.1985,
            #'t1_66':   
            -0.3330,
            #'t1_77':   
            0.3190,
            #'t1_88':   
            -0.5837,
            #'t1_99':   
            -0.2399,
            #'t1_1010':   
            1.0470,
            #'t1_1111':   
            0.0029,
            #'t1_35':   
            -0.0912,
            #'t1_68':   
            0.4233,
            #'t1_911':   
            -0.0377,
            #'t1_12':   
            -0.2321,
            #'t1_34':   
            -0.0797,
            #'t1_45':   
            -0.0920,
            #'t1_67':   
            -0.1250,
            #'t1_78':   
            -0.2456,
            #'t1_910':   
            0.1857,
            #'t1_1011':   
            -0.1027,

            #'t5_41':   
            -0.7744,
            #'t5_32':   
            -1.4014,
            #'t5_52':   
            2.0858,
            #'t5_96':   
            -0.8998,
            #'t5_116':   
            -0.9044,
            #'t5_107':   
            1.4030,
            #'t5_98':   
            -0.8548,
            #'t5_118':   
            0.5711,

            #'t6_96':   
            -0.0676,
            #'t6_116':   
            -0.1608,
            #'t6_98':   
            -0.2618,
            #'t6_118':   
            -0.2424,

            #SO
            #'W':
            0.2874,
            #'Se'
            0.2470,

            #'offset
            -0.73,
                ],
        }

#Names of independent parameters of the model
list_names_all = [
            'e1', 
            'e3',   
            'e4',   
            'e6',   
            'e7',   
            'e9',   
            'e10',  
            't1_11',
            't1_22',   
            't1_33',   
            't1_44',   
            't1_55',   
            't1_66',   
            't1_77',   
            't1_88',   
            't1_99',   
            't1_1010',   
            't1_1111',
            't1_35',   
            't1_68',   
            't1_911',   
            't1_12',
            't1_34',   
            't1_45',   
            't1_67',   
            't1_78',   
            't1_910',   
            't1_1011',   
            't5_41',   
            't5_32',   
            't5_52',   
            't5_96',   
            't5_116',   
            't5_107',   
            't5_98',   
            't5_118',   
            't6_96',   
            't6_116',   
            't6_98',   
            't6_118',          
            'L_W',
            'L_S',
            'offset'
            ]

#Spin-Orbit Hamiltonian
#It is k-independent so depends only on \lambda of the materials in the TMD -> 2 parameters for each layer
dic_params_SO = {
        'W': 0.2874,
        'S': 0.0556,
        'Se': 0.2470,
        }

#Monolayer lattice lengths, in Angstrom
dic_params_a_mono = {
        'WS2': 3.18,
        'WSe2': 3.32,
        }
#Moirè lattice length of bilayer, in Angstrom
dic_a_Moire = { 'WS2/WSe2':79.8,
            'WSe2/WS2':79.8,
       }

#Moirè potential of bilayer
#Different at Gamma (d_z^2 orbital) -> first two parameters, and K (d_xy orbitals) -> last two parameters
#Gamma point values from paper "G valley TMD moirè bands" (first in eV, second in radiants)
#K point values from Louk's paper (first in eV, second in radiants)
dic_params_V = {'WSe2/WS2':[0.0335,np.pi, 7.7*1e-3, -106*2*np.pi/360],
            }

