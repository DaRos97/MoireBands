Compare side bands distances.

There are many steps to make a good comparison:
- with `1_cut_image.py` we refine the size of S3 and S11 experimental images.
- with `2_mass.py` we fit the WSe2 band with a parabula to get the mass and the offset.
- with `3_distance.py` we compute the side band distances by extracting the intensities at given momentum and energy.
- with `4_side_bands.py` we finaly compare analytic, numeric and experimental results.
- with `5_results.py` we plot the estimated twist angle as a function of V for various phi, both samples.
