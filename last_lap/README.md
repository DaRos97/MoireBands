# Codes
We care about the KGK image. Use 11-band model, with interlayer and Moire.
1 - See DFT monolayer of tight binding, stay close to DFT and fit a bit better the experimental bands.
    The stay close is important to have good band content.
2 - Without Moire replicas, add interlayer coupling between `p_z` orbitals to get the lower band (WS2 TVB) flatter and the higher band (WSe2 TVB) sharper.
    Use the simple model values found for 'a' and 'b' and 'c'.
3 - Add Moire replicas with Moire potential amplitude and phase given by the simple model S11 fit.
    This should give the S11 image KGK.
4 - Compute CEMs around Gamma and K.
    Check for 3/6-fold symmetry.

--- Talk with Louk about changing the interlayer coupling between `p_z` orbitals to try to find S3 -> interlayer between Moire replicas.
--- Don't really wanna do this but check orbital content around Gamma and see if the k dependence is 3-fold -> this might indicate that evaluating the matrix elements is worth it.

# Presentation 10/10/2024
- Figure out tight binding minimization parameters
- Compute some images with also these parameter -> should be better at K, about the same at Gamma
- Check rotation angle of moire reciprocal lattice vectors

