#Code
Use for interlayer just between p_z(even) since they are p_z orbitals and is the most occupied orbital at Gamma.
The interlayer is parametrized by constants a,b and c as
H_int = (E(p_z)_WSe2    -a+b*k^2    )
        (-a+b*k^2       E(p_z)_WS2+c)
***
In order to make the lower band more decaying close to gamma we can add a factor of d=-0.2 decreasing the p_x(odd) orbital -> better looking.
****
Best parameters by eye are (global offset of 0.5 eV)
    DFT: a=1, b=0.7, c=0.75, d = -0.2
    min: a=1.1, b=0.7, c=1, d=0
Parameters might change for different optimal parameters because of different orbital content.
****
If we want to add also interlayer close to K point we need to consider p_x(even), but maybe not needed.

#Import tb values
- get_res_1.py
Imports the results from 1_tight_binding/. Imports both DFT and chosen minimi9zatipon result.

