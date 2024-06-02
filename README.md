# lid-driven-cavity-gpu
This repository contains the project that implements the lid driven cavity problem 
using the GPU. It also compares the performance to the CPU version.

It's a project for the subject "Matematika z računalnikom" at the Faculty of Mathematics and Physics 
in Ljubljana.

# Installation instructions
## Requirements
- Cuda 12
- Medusa
- HDF5
- cuDSS
## Installation
The project is meant to be run on a specific server and as such might not be the easiest to compile in general.

First the [medusa library](https://gitlab.com/e62Lab/medusa) has to be installed into the home directory,
which is where the program currently assumes it is located. During the installation of the medusa library 
HDF5 has to be installed, which is also a dependency for this project.

Next [cuDSS](https://developer.nvidia.com/cudss) is installed by downloading the tarbal from the website and exporting it into the 
home directory.

Afterwards the project can be built using the `cmake` and `make` commands. For example from the `build` directory with
```bash 
cmake ..
make
```
The compiled files with be placed in the `bin` directory from where they can be run as
```bash
lidDriven ../params/chanelFlow.xml
```
for the pressure projection method and 
```bash
lidDrivenACM ../params/chanelFlowACM.xml
lidDrivenMatrixACM ../params/chanelFlowACM.xml
lidDrivenMatrixACMGPU ../params/chanelFlowACM.xml
```
for the artificial compressibility method where the binaries implement the ACM method,
ACM where the matrices are initialized at the start and the ACM method that uses the GPU respectively.

## Project structure
- The `benchmark` folder consists of jupyter notebook which runs the various benchmarks and plots the execution times alongside some 
correctness checks. It also contains the report containing the final results of the project, which is effectively an extension 
of the final report for the university subject located in the `report` folder.
- The `bin` folder contains the executables for the project after they are built.
- The `build` directory is meant to contain all build files.
- The `params` folder contains the base parameter files for the programs. From there various runtime parameters can be set as 
well as the choice of the solver for the pressure projection method.
- The `report` repository contains the final report for the university subject. 
- The `scripts` folder contains a sample script of using the HDF5 library with this project. It was provided at the start of the project.
- The `src` folder contains all the source code for the project. It can be broken up into two parts:
    + For the pressure correction method the `lidDriven.cpp` file contains the base code which can
    based on the parameter file choose a solver implemented in the `src/solvers` folder. The available 
    solvers are: The wrapper of the `Eigen::SparseLU` solver, cuSolver based QR solver, cuDSS based LUD solver and 
    a cuSolver based RF solver which allows for reusing the same decomposition of the matrix multiple times.
    + For the ACM method there are three C++ files. The first two `lidDrivenACM` and `lidDrivenMatrixACM` were provided 
    as base CPU implementations, while the `lidDrivenMatrixACMGPU` file contains the code for the version which uses the GPU
    to implement the method.
