"""
Convert yearly NetCDF velocity fields to MATLAB . mat format. 

The transition-matrix and trajectory scripts use velocity files stored as 
MATLAB arrays with variable names "u", "v", "x", "y", and "t". 
This script converts NetCDF files in "DATA_DIR" to that format. 

The velocity arrays are saved with shape "(lat, lon, time)". 

"""

from pathlib import Path

from netCDF4 import Dataset
import numpy as np
from scipy.io import savemat
from config import DATA_DIR

def convert_nc_to_mat(nc_path: Path, overwrite: bool = False) -> Path:
    """
Convert one NetCDF file to MATLAB .mat format.

Parameters
----------
nc_path : Path
    Path to the input .nc file.
overwrite : bool, optional
    If True, overwrite an existing output .mat file.

Returns
-------
Path
    Path to the created .mat file.
"""
    if not nc_path.exists():
        raise FileNotFoundError(f"Input file not found: {nc_path}")

    mat_path = nc_path.with_name(f"{nc_path.stem}.mat")

    if mat_path.exists() and not overwrite:
        print(f"[SKIP] {mat_path.name} already exists")
        return mat_path
    
    with Dataset(nc_path) as ds:
        # Read variables from the NetCDF file
        t_s1 = ds.variables["time"][:]        # time (NT,)
        y1 = ds.variables["latitude"][:]      # Latiude (NY,)
        x1 = ds.variables["longitude"][:]     # Longitude (NX,)
        u1 = ds.variables["ugos"][:]          # Velocity (lon, lat, time)
        v1 = ds.variables["vgos"][:]          # Velocity (lon, lat, time)
        
    # Rearrange velocity arrays from (lon, lat, time) to (lat, lon, time)
    u = np.transpose(u1, (1, 2, 0))
    v = np.transpose(v1, (1, 2, 0))

    # Convert to float arrays of size (1,N)
    t = np.asarray(t_s1, dtype=float).reshape(1, -1)
    x = np.asarray(x1, dtype=float).reshape(1, -1)
    y = np.asarray(y1, dtype=float).reshape(1, -1)
    

    # Save processed arrays to .mat
    savemat(mat_path, {
        "u": u,
        "v": v,
        "x": x,
        "y": y,
        "t": t,
    })

    print(f"[OK] Converted {nc_path.name} -> {mat_path.name}")
    return mat_path



def main() -> None:
    # Find all NetCDF files in DATA_DIR and sort alphabetically
    nc_files = sorted(DATA_DIR.glob("*.nc"))
    
    if not nc_files:
        print(f"[INFO] No .nc files found in {DATA_DIR}")
        return
    
    # Loop through each NetCDF file and convert it 
    for nc_path in nc_files:
            convert_nc_to_mat(nc_path, overwrite=True)

if __name__ == "__main__":
    main()
    