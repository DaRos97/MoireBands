# Codes

# 1 - tight binding model

# 2 - interlayer coupling
Use for interlayer just between p_z(even) since they are p_z orbitals and is the most occupied orbital at Gamma.
The interlayer is parametrized by constants a,b and c as
H_int = (E(p_z)_WSe2    -a+b*k^2    )
        (-a+b*k^2       E(p_z)_WS2+c)
****
If we want to add also interlayer close to K point we need to consider p_x(even), but maybe not needed.
****
Best parameters by eye are a=1, b=0.7, c=0.75, mu(global offset) = 0.25
In order to make the lower band more decaying close to gamma by adding a factor of d=1 decreasing the p_x(odd) orbital -> better looking.

# 3 - KGK image
