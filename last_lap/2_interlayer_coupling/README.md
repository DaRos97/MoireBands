#Code
Use for interlayer just between p_z(even) since they are p_z orbitals and is the most occupied orbital at Gamma.
The interlayer is parametrized by constants a,b and c as
H_int = (E(p_z)_WSe2    -a+b*k^2    )
        (-a+b*k^2       E(p_z)_WS2+c)
****
If we want to add also interlayer close to K point we need to consider p_x(even), but maybe not needed.
****
Best parameters by eye are a=1, b=0.7, c=0.75 for DFT.
Parameters might change for different optimal parameters because of different orbital content.
***
In order to make the lower band more decaying close to gamma we can add a factor of d=0.2 decreasing the p_x(odd) orbital -> better looking.
