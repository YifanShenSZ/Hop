# Developers Notes for Hop
We develope this package based on the LTS pytorch build 1.8.2, so using hop along with libtorch 1.8.2 is recommended

Since libtorch 1.8.2 does not support complex number, we work with real tensors of shape (..., 2)

## Todo
The super-exchange problem should exist in anniline photodissociation:
* After anniline travel back to min-A1 from the **h** direction of mex-A1-B1, it should be able to hop onto B2 then B1 then try dissociating again
* The hot vibration should along NH2 rotation, which is also the A1-B2 nonadiabatic coupling direction
* However, B2 state might be a little bit too high in energy. More possibly, it should super-exchange to B1 at somewhere between min-A1 and sad-A1

GFSH is proposed for super-exchange, but it does not provide a way to rescale momentum in multi-dimensional case. So we need to either extend GFSH or propose a new method

Currently I pictured two candidates:
1. MFSH: minimize the sum square of fluxes, but the momentum rescaling is as vague as GFSH
2. MSTSH: minimum spanning tree from active state, may do the momentum rescaling by the average nonadiabatic coupling along the tree path