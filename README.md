# Apolemia Transport TPT

## Overview
This repository contains code for a master's thesis on Lagrangian ocean transport in the Norwegian Sea. The workflow constructs finite-time, time-inhomogeneous transition matrices (titm) and applies finite-time Transition Path Theory (TPT) to study transport diagnostics associated with Apolemia bloom years.


## Repository structure
```text
project/
└─ src/
   ├─ config.py
   ├─ convert_nc_to_mat.py
   ├─ build_titm.py
   ├─ diagnostics_core.py
   ├─ diagnostics_plots.py
   ├─ run_tpt_diagnostics.py
   ├─ data/
   │  ├─ *.nc
   │  └─ *.mat
   ├─ Output/
   │  ├─ titm1000/
   │  └─ Run<timestamp>/
   │     ├─ flux_fields/
   │     ├─ metrics/
   │     ├─ multi_step/
   │     └─ plots/   
   └─ third_party/
      ├─ pytpt_finite_helfmann.py
      ├─ Tbarrier_hallergroup.py
      ├─ pygtm_miron
         ├─ dataset.py
         ├─ matrix.py
         ├─ physical.py
         └─ tools.py
```

## Data
Surface transport is modeled using altimetry-derived geostrophic velocity fields from the Copernicus Marine Service for the period 1996–2024. 

Dataset: 
SEALEVEL_GLO_PHY_L4_MY_008_047
Download link: https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/download?dataset=cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_202411

When downloading the data, select:
- Daily fields
- Surface geostrophic eastward water velocity (ugos, m/s)
- Surface geostrophic northward water velocity (ugos [m/s])

Spatial domain:
- West: -18
- East: 23
- South: 48
- North: 74

Temporal coverage used in this thesis:
- 1996-12-01 to 2024-12-31

The TITM constructon requires overlapping one-year netCDF files. For example: 
- "NAO1997.nc" contains data from 1996-12-01 to 1997-12-31 and is used to construct the 1997 transition matrices
- "NAO1998.nc" contains data from 1997-12-01 to 1998-12-31 and is used to construct the 1998 transition matrices

This overlap is required because the 14-day trajectory integration preceeding ISO week 2 may begin in the previous calender year. 

All downloaded ".nc" files should be placed in "src/data/". 

## Workflow
The code is intended to be run in the following order: 

1. Review and edit "src/config.py" to set years, domain, source region, targets, gates, and output options. 
2. Download the raw netCDF velocity data and save the files in "src/data/"
3. Run "convert_nc_to_mat.py" to convert all ".nc" files in "src/data/" into MATLAB ".mat" files.
4. Run "build_titm.py" to construct the yearly sequence of augmented time-inhomogeneous transition matrices "P_ains_<year>.pkl"
5. Run "run_tpt_diagnostics.py" to compute transport anomaly diagnostics and associated plots. 

## Files

### "config.py"
Shared configuration file containing paths, domain settings, year lists, source and gate definition, and output folder definitions. 

### "convert_nc_to_mat.py"
Converts all NetCDF files in "src/data/" into MATLAB ".mat" files used by the TITM builder. 

### "build_titm.py"
Builds yearly sequence of augmented time-inhomogeneous transition matrices "P_ains_<year>.pkl" from the converted ".mat" files. 

### "diagnostics_core.py"
Contains the core transport diagnostics routines, including transition path theory application, climatological weighting distribution, leave-one-out scheme, whole-system transport distance, total transition rate, and gate share. 

### "diagnostics_plots.py"
Contains all plotting routines used in the diagnostics stage. 

### "run_tpt_diagnostics.py"
Main diagnostics runner. Uses the saved yearly TITMs to compute transport diagnostics and generate plots and summary outputs. 

## Third-party code
The folder "src/third_party/" contains adapted third-party code used by the workflow. 

### "pytpt_finite_helfmann.py"
Finite-time Transition Path Theory implementation used in the diagnostics calculations.

### "Tbarrier_hallergroup.py"
Interpolation, velocity conversion, and trajectory integration utilitites used in the TITM construction.

### "pygtm_miron/"
Contains adapted components from the pyGTM framework: 
- "dataset.py"
- "matrix.py"
- "physical.py"
- "tools.py"

These files are used for trajectory handling, transition matrix construction, and spatial discretization.

## Outputs
All generated files are written to "src/Output". 

- "src/Output/titm<m>/" contains the yearly sequence of augmented transition matrices "P_ains_<year>.pl"
- "src/Output/titm<m>/titm_tests/" contains tests performed on the augmented transition matrices.
- "src/Output/Run<timestamp>/plots/" contains figures from the diagnostic stages
- "src/Output/Run<timestamp>/metrics/" containes saved per-target transport metrics
- "src/Output/Run<timestamp>/multi_step/" contains saved full-window transport operators used in the transport diagnostics.

## Requirements
The code was written for Python 3.12 and requires the following Python packages: 
- numpy
- pandas
- scipy
- matplotlib
- cartopy
- netCDF4
- scikit-learn

The repository also includes the required third-party research code in "src/third_party/", so these files do not need to be installed separately as packages. 

## License and attribution
This repository includes adapted code from the following third-party research projects: 
- "pytpt_finite_helfmann.py" is adapted from "pytpt" project: https://github.com/LuzieH/pytpt
- "Tbarrier_hallergroup.py" is adapted from "Tbarrier" project: https://github.com/EncinasBartos/TBarrier
- "pygtm_miron/" contains adapted components from the "pygtm" project: https://github.com/philippemiron/pygtm

Please consult the original repositories for their licenses and attribution requirements. 
