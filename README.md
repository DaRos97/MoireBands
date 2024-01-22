# General structure

Folders 0,1,... have the actual code in the order intended to be executed.

# Old code

Folders eleven/three band model are the middle steps.

# What has been done

- Three band model: 
    - KGK cut and CEMs
    Problem:    Crossing of bands b/w Gamma and K -> TB model not good
- Eleven band model:    
    - Fitting of monolayers to the 44 parameters of the model (problem in GG data..) -> 1_monolayer_fitting
    - KGK cut
    Problem: no C3 symmetry around Gamma
- Simple model:
    0_simple_model:
        - Use simple parabolic dispersion and add INTERLAYER coupling -> fit around Gamma.
        - Minimization with GG data to get Moire potential intesity of side band around Gamma -> found intensity too high (S3).
        - Same around K -> works better
        - Tryed sqrt of intensity -> not physical
    3_new_simple_model:
        - Fit simple model with also Moirè potential to be more accurate
        - Minimization with GG data varying V,A_M and spread_E -> got good agreement with S11 around Gamma.
    4_CEM_K:
        - Compute CEM around K using parabolic dispersion from monolayer data.
        - Add interlayer to check differences between Parallel and Anti-Parallel alignment of BZs -> want to see different symmetry

Open problems:
    - C3 symmetry around Gamma
    - S3 side band around Gamma
    - Matrix elements ? 

# Last lap
