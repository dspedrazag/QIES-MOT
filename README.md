# MOT Magnetic Field Data and Simulations

This repository contains the experimental magnetic-field measurements and the simulation tools used to characterize the MOT coils.

## MOT Data

The `MOT Data` folder contains the experimental measurements of the magnetic field.

Inside this folder, the notebook:

`Compared-graphs.ipynb`

contains the analysis and plots of the experimental data. In particular, it includes:

* Experimental magnetic-field measurements.
* Experimental (B_z) and (B_x) profiles.
* Calculation and visualization of the corresponding magnetic-field gradients.

## Magnetic Field Simulations

The notebooks located outside the `MOT Data` folder, corresponding to the **DeMille** and **Pedrozo** gradient calculations, contain the coil parameters and constraints used to simulate the magnetic field.

The file:

`ArbitraryPointBField.py`

contains the main functions used for the magnetic-field simulations, including the calculation of the field produced by the MOT coil geometry.

## Experimental Data vs. Simulation

The notebook:

`ExpvsSimulation.ipynb`

compares the magnetic-field simulation with the experimental measurements. It contains plots showing the simulated and measured magnetic fields together, allowing the agreement between the model and the experimental data to be evaluated.
