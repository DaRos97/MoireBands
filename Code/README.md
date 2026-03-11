# Structure

The code here presented is for the analysis of ARPES spectra in TMD mono-layers and hetero-bilayers.

## Code

It is divided into `monolayer_v*/` and `bilayer_v*/` codes. 

### Monolayer
`v_2.0`
The analysis starts from the fitting of the monolayer tight-binding parameters.
`main.py` for the analysis and `sortResults.py` for the post-processing.
`visualize_ARPES_data.py` to see what data is being fitted.

### Bilayer
`v_2.0`
REMEMBER to update the tight binding parameters to the newest/final ones.
edc fitting at G and K with `edc.py` to extract the moiré potential and interlayer coupling.

Also the LDOS is computed to compare with STM.

## Figures
REMEMBER to update the parameters to the newest/final ones:
    - monolayers tight binding
    - interlayer coupling
    - moiré potential

- `fig_interlayer/` for the figure of the bilayer with interlayer coupling and without moiré potential
- `fig_moire/` for the full figure with moiré potential and broadening of intensities
- `fig_monolayer/` for the figure of the monolayer comparing ARPES and DFT-derived parameters
- `fig_edc/` for the results of the edc and comparison of intesity cuts with ARPES
