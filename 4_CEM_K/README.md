# What's what

Here we compute CEM around K. We do it in two different ways: with and without interlayer.
Why?
Because in case of 0° rotation, the two layers have K on top of K (referred to as parallel 'P') but in case of 60° rotation K is on 
top of K' (anti-parallel 'AP'). In both cases we expect C3 symmetry because of the BZ shape (K to K/K' or to Gamma). For P there is 
in principle more interlayer coupling since the spin is locked to the valley and the two valleys have the same spin. Instead for AP
the interlayer should be reduced.

In this code we use the parabolic single orbital ('simple') model, with and without interlayer coupling.

We use a Hamiltonian of the form:
H = (-k^2/2m1       t(k)            )       with t(k) = -a(1-bk^2) and a diagonal energy shift mu.
    (t*(k)          -k^2/2m2 - c    )

In the P case we fit the image using a=b=0. In the AP case we include them.

We expect to reproduce S11 in the P case and S3 in the AP case (right?).
