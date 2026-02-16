#Description
Use for interlayer just between pz(even) since they are pz orbitals and is the most occupied orbital at Gamma, where the interlayer coupling should be most intense.
The interlayer is parametrized by constants a,b and c as
Hi =    (E(pz) WSe2     t(k)             )
        (t(k)           E(pz) WS2 + c   )

t(k) can be:
    - C6 symmetric   -> -a+b\sum_{j=1}^6\exp{i k\cdot aj} = -a + 2(cos(kx a)+cos(kx/2 a)cos(sqrt(3)/2 ky a))    #the 'a' in cosines is the lattice constant of WSe2
    - C3 symmetric   -> b'\sum_{j=1}^3\exp{i k\cdot deltaj} #which now is complex and the delta are vectors connecting S and Se

We consider the interlayer parameters as (a,b,c,offset). 
In C3, a=0. 

Hi couples pz^e with pz^e, equally in the two spin sectors. 

Best parameters by eye are in interlayer.py for DFT/fit and all types of interlayer.

Parameters of fit might change for different fit parameters because of different orbital content.

Parameters might also differ for S11 and S3.

#Code
- `copy_fit_pars.sh` imports fit results from step 1.
- `interlayer.py` & `functions_interlayer.py`         #Compute the best interlayer -> by eye.
- `hand_comparison.sh` computes repetedly the script to get many images and then comare.

# Results
The best parameters for fit and interlayer types are stored in the `interlayer.py` script.
Figures/temp/ is used for storing the temporary images of when loking for best parameters.
Figures/ has the final images.





