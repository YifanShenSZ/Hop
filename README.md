# Hop
Surface hopping with diabatic potential energy matrix

Available electronic wave function propagation:
1. electronic schrodinger equation
2. phase correction

Available surface hopping probability:
1. none (no hopping, just adiabatic molecular dynamics)
2. fewest switches
3. global flux

## Usage
Link to library `hop` then grab a hopper

You will need to write your own control loop, since the hopper API only lets you propagate one time step per call (similar to pytorch optimizer)

Library `initial` provides a useful utility to generate nuclear initial coordinate and momentum

## Test models
Tully 1st model: simple avoided crossing `SAC-Tully1`

Tully 2nd model: dual avoided crossing `DAC-Tully2`

Tully 3rd model: extended coupling with reflection `ECR-Tully3`

## Reference
> 1. N. Shenvi, J. Subotnik, W. Yang 2011 JCP
> 2. J. Tully 1990 J. Chem. Phys.
> 3. L. Wang, D. Trivedi, O. Prezhdo 2014 JCTC
