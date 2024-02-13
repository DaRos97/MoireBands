#Description
Use for interlayer just between pz(even) since they are pz orbitals and is the most occupied orbital at Gamma, where the interlayer coupling should be most intense.
The interlayer is parametrized by constants a,b and c as
Hi =    (E(pz) WSe2     t(k)             )
        (t(k)           E(pz) WS2 + c   )

t(k) can be:
    - U1 symmetric -> -a+b|k|^2
    - C6 symmetric   -> -a+b\sum_{j=1}^6\exp{i k\cdot aj} = -a + 2(cos(kx a)+cos(kx/2 a)cos(sqrt(3)/2 ky a))    #the 'a' in cosines is the lattice constant of WSe2 or WS2
    - C3 symmetric   -> b'\sum_{j=1}^3\exp{i k\cdot deltaj} which now is complex and the delta are vectors connecting S and Se
We consider the interlayer parameters as (a,b,c,offset). 
In C3, a=0. 
In C6 and C3, c=0?

Hi couples pz^e with pz^e. 
We can use the same to couple pz^o with pz^o. 
Also we can couple pz^e with pz^o and vice-versa BUT changing a sign. WHICH?

Best parameters by eye are in coupling_interlayer.py for DFT/fit and all types.

Parameters of fit might change for different fit parameters because of different orbital content.


#Code
- get_res_1.py      Imports the results from 1_tight_binding/. Imports both DFT and chosen minimizatipon result.
- coupling_interlayer.py & functions.py         Compute the best interlayer -> by eye.


