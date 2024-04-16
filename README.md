# lid-driven-cavity-gpu
Simulation of the lid driven cavity on the GPU. 

A project for the FMF subject MATRAČ.

# Installation instructions
## requirements
- Cuda 12
- Medusa
- cuDSS
## installation
From gitlab clone the [medusa library](https://gitlab.com/e62Lab/medusa) into the home directory,
which is where the program currently assumes it is located.

Download the tarball for the [cuDSS](https://developer.nvidia.com/cudss) and export it into the home directory.

Afterwards the project can be built using the `cmake` and `make` commands. 