#Description
Use for interlayer just between pz(even) since they are pz orbitals and is the most occupied orbital at Gamma, where the interlayer coupling should be most intense.
The interlayer is parametrized by constants a,b and c as
Hi =    (E(pz) WSe2     t(k)             )
        (t(k)           E(pz) WS2 + c   )

t(k) can be:
    - U1 symmetric -> -a+b|k|^2
    - C6 symmetric   -> -a+b\sum_{j=1}^6\exp{i k\cdot aj} = -a + 2(cos(kx a)+cos(kx/2 a)cos(sqrt(3)/2 ky a))    #the 'a' in cosines is the lattice constant of WSe2 or WS2
    - C3 symmetric   -> b'\sum_{j=1}^3\exp{i k\cdot deltaj} #which now is complex and the delta are vectors connecting S and Se

We consider the interlayer parameters as (a,b,c,offset). 
In C3, a=0. 

Hi couples pz^e with pz^e, equally in the two spin sectors. 

Best parameters by eye are in coupling_interlayer.py for DFT/fit and all types of interlayer.

Parameters of fit might change for different fit parameters because of different orbital content.

Parameters might also differ for S11 and S3.

#Code
- `get_res_1.py`      #Imports the results from 1_tight_binding/. Imports both DFT and chosen minimizatipon result.
- `coupling_interlayer.py` & `functions.py`         #Compute the best interlayer -> by eye.

# Results
The best parameters for fit and interlayer types are stored in the coupling_interlayer.py script. 
temp/ is used for storing the temporary images of when loking for best parameters.
figures/ has the final images.

# Best values (a,b,c,offset)
## DFT
    no: (0,0,0,-0.5)
    U1: (1,0.7,0.7,-0.5)
    C6: (0.1,0.29,0.65,-0.5)
    C3: (0,0.33,0.75,-0.5)
## fit
    no: (0,0,0,-0.48)
    U1: (1,0.9,0.88,-0.48)
    C6: (0.15,0.32,0.75,-0.48)
    C3: (0,0.35,0.8,-0.48)
