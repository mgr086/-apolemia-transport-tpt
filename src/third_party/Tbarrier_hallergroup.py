    # =============================================================================
#                               TBarrier
# =============================================================================
import numpy as np
from scipy.interpolate import RectBivariateSpline as RBS

def convert_meters_per_second_to_deg_per_day(X, Y, U_ms, V_ms):
    
    '''
    Converts units of velocity from m/s to deg/day. The units of the velocity field must 
    match the units of the grid coordinates and time.
    
    Parameters:
        X:       array(Ny, Nx), X-meshgrid.
        Y:       array(Ny, Nx), Y-meshgrid.
        U_ms:    array(Ny, Nx, Nt), x-component of velocity field in m/s
        V_ms:    array(Ny, Nx, Nt), y-component of velocity field in m/s
         
    Returns:
        U_degday:    array(Ny, Nx, Nt), x-component of velocity field in deg/day
        V_degday:    array(Ny, Nx, Nt), y-component of velocity field in deg/day
    '''
        
    # Velocity field
    U_degday, V_degday = np.nan*U_ms.copy(), np.nan*V_ms.copy()
    
    # Radius of the earth
    earthRadius = 6371*(10**3)
    
    # Iterate over grid
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            U_degday[i, j, :] = (U_ms[i, j, :] / (np.cos(Y[i,j]*(np.pi/180))*(earthRadius)))*180*3600*24/np.pi
            V_degday[i, j, :] = (V_ms[i, j, :] / earthRadius)*180*3600*24/np.pi
        
    return U_degday, V_degday


def interpolant_unsteady(X, Y, U, V, method = "cubic"):
    '''
    Unsteady wrapper for scipy.interpolate.RectBivariateSpline. Creates a list of interpolators for u and v velocities
    
    Parameters:
        X: array (Ny, Nx), X-meshgrid
        Y: array (Ny, Nx), Y-meshgrid
        U: array (Ny, Nx, Nt), U velocity
        V: array (Ny, Nx, Nt), V velocity
        method: Method for interpolation. Default is 'cubic', can be 'linear'
        
    Returns:
        Interpolant: list (2,), U and V  interpolators
    '''
    # Cubic interpolation
    if method == "cubic":
                
        kx = 3
        ky = 3
               
    # linear interpolation
    elif method == "linear":
            
        kx = 1
        ky = 1  
            
    # define u, v interpolants
    Interpolant = [[], []]
                    
    for j in range(U.shape[2]):
                
        Interpolant[0].append(RBS(Y[:,0], X[0,:], U[:,:,j], kx=kx, ky=ky))
        Interpolant[1].append(RBS(Y[:,0], X[0,:], V[:,:,j], kx=kx, ky=ky))
    
    return Interpolant


def velocity(t, x, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data):
    '''
    Evaluate the interpolated velocity field over the specified spatial locations at the specified time.
    
    Parameters:
        t:              float,  time instant  
        x:              array (2,Npoints),  array of ICs
        X:              array (NY, NX)  X-meshgrid of data domain
        Y:              array (NY, NX)  Y-meshgrid of data domain
        Interpolant_u:  Interpolant object for u(x, t)
        Interpolant_v:  Interpolant object for v(x, t)
        periodic:       list of 3 bools, periodic[i] is True if the flow is periodic in the ith coordinate. Time is i=3.
        bool_unsteady:  bool, specifies if velocity field is unsteady/steady
        time_data:      array(1, NT) time of velocity data
    Returns:

        vel:            array(2,Npoints), velocities, vel[0,:] --> x-coordinate of velocity, vel[1,:] --> y-coordinate of velocity
    '''
    x_eval = x.copy()
    
    # check if periodic in x
    if periodic[0]:
        
        x_eval[0,:] = (x[0,:]-X[0, 0])%(X[0, -1]-X[0, 0])+X[0, 0]
    
    # check if periodic in y
    if periodic[1]:
        
        x_eval[1,:] = (x[1,:]-Y[0, 0])%(Y[-1, 0]-Y[0, 0])+Y[0, 0]
        
    if periodic[2]:
        
        t = t%(time_data[0, -1]-time_data[0, 0])+time_data[0, 0]
    
    dt_data = time_data[0,1]-time_data[0,0]
    
    # Unsteady case
    if bool_unsteady:

        k = int((t-time_data[0, 0])/dt_data)
    
        # evaluate velocity field at time t_eval
        if k >= len(Interpolant_u)-1:
            
            u = Interpolant_u[-1](x_eval[1,:], x_eval[0,:], grid = False)
            v = Interpolant_v[-1](x_eval[1,:], x_eval[0,:], grid = False)
            
        else: 
    
            ui = Interpolant_u[k](x_eval[1,:], x_eval[0,:], grid = False)
            uf = Interpolant_u[k+1](x_eval[1,:], x_eval[0,:], grid = False)
            u = ((k+1)*dt_data-(t-time_data[0,0]))/dt_data*ui + ((t-time_data[0,0])-k*dt_data)/dt_data*uf

            vi = Interpolant_v[k](x_eval[1,:], x_eval[0,:], grid = False)
            vf = Interpolant_v[k+1](x_eval[1,:], x_eval[0,:], grid = False)
            v = ((k+1)*dt_data-(t-time_data[0,0]))/dt_data*vi + ((t-time_data[0,0])-k*dt_data)/dt_data*vf
        
    # Steady case        
    elif bool_unsteady == False:
            
        u = Interpolant_u(x_eval[1,:], x_eval[0,:], grid = False)
        v = Interpolant_v(x_eval[1,:], x_eval[0,:], grid = False)
        
    vel = np.array([u, v])
    
    return vel

def integration_dFdt(time, x, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data, verbose = False):
    '''
    Wrapper for RK4_step(). Advances the flow field given by u, v velocities, starting from given initial conditions. 
    The initial conditions can be specified as an array. 
    
    Parameters:
        time:          array (Nt,),  time array  
        x:             array (2, Npoints),  array of ICs (#Npoints = Number of initial conditions)
        X:             array (NY, NX),  X-meshgrid (of complete data domain)
        Y:             array (NY, NX),  Y-meshgrid (of complete data domain)
        Interpolant_u: Interpolant object for u(x, t)
        Interpolant_v: Interpolant object for v(x, t)
        periodic:      list of 3 bools, periodic[i] is True if the flow is periodic in the ith coordinate. Time is i=3.
        bool_unsteady: bool, specifies if velocity field is unsteady/steady
        time_data:     array(1,NT), time data
        verbose:       bool, if True, function reports progress at every 100th iteration
    
    Returns:
        Fmap:          array (Nt, 2, Npoints), integrated trajectory (=flow map)
        dFdt:          array (Nt-1,2, Npoints), velocity along trajectories (=time derivative of flow map) 
    '''
    # reshape x
    x = x.reshape(2, -1)

    # Initialize arrays for flow map and derivative of flow map
    Fmap = np.zeros((len(time), 2, x.shape[1]))
    dFdt = np.zeros((len(time)-1, 2, x.shape[1]))
    
    # Step-size
    dt = time[1]-time[0]
    
    counter = 0

    # initial conditions
    Fmap[counter,:,:] = x
    
    # Runge Kutta 4th order integration with fixed step size dt
    for counter, t in enumerate(time[:-1]):
        if verbose:
            if counter%100 == 0:
                print('Percentage completed: ', np.around((t-time[0])/(time[-1]-time[0])*100, 4))
        
        Fmap[counter+1,:, :], dFdt[counter,:,:] = RK4_step(t, Fmap[counter,:, :], dt, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data)
    
        # check if periodic in x
        #if periodic[0]:
        
        #    Fmap[counter+1,0,:] = (Fmap[counter+1, 0,:]-X[0,0])%(X[0, -1]-X[0, 0])+X[0,0]
    
        # check if periodic in y
        #if periodic[1]:
        
        #    Fmap[counter+1,1,:] = (Fmap[counter+1, 1, :]-Y[0,0])%(Y[-1, 0]-Y[0, 0])+Y[0,0]
    
        counter += 1

    return Fmap, dFdt


def integration_dFdt_final(time, x, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data, verbose = False):
    '''
    Wrapper for RK4_step(). Advances the flow field given by u, v velocities, starting from given initial conditions. 
    The initial conditions can be specified as an array. 
    
    Parameters:
        time:          array (Nt,),  time array  
        x:             array (2, Npoints),  array of ICs (#Npoints = Number of initial conditions)
        X:             array (NY, NX),  X-meshgrid (of complete data domain)
        Y:             array (NY, NX),  Y-meshgrid (of complete data domain)
        Interpolant_u: Interpolant object for u(x, t)
        Interpolant_v: Interpolant object for v(x, t)
        periodic:      list of 3 bools, periodic[i] is True if the flow is periodic in the ith coordinate. Time is i=3.
        bool_unsteady: bool, specifies if velocity field is unsteady/steady
        time_data:     array(1,NT), time data
        verbose:       bool, if True, function reports progress at every 100th iteration
    
    Returns:
        y:     Returns only final positions. array (2, Npoints), integrated trajectory
    '''
    
    x = x.reshape(2, -1)
    
    
    # Step size
    dt = time[1] - time[0]
    y = x  # current positions

    for counter, t in enumerate(time[:-1]):
        if verbose and counter % 100 == 0:
            print('Percentage completed: ', np.around((t-time[0])/(time[-1]-time[0])*100, 4))

        y, _ = RK4_step(t, y, dt, X, Y, Interpolant_u, Interpolant_v,
                        periodic, bool_unsteady, time_data)

    return y


def RK4_step(t, x1, dt, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data):
    '''
    Advances the flow field by a single step given by u, v velocities, starting from given initial conditions. 
    The initial conditions can be specified as an array. 
    
    Parameters:
        time:           array (Nt,),  time array  
        x:              array (2, Npoints),  array of currents positions
        X:              array (NY, NX)  X-meshgrid
        Y:              array (NY, NX)  Y-meshgrid 
        Interpolant_u:  Interpolant object for u(x, t)
        Interpolant_v:  Interpolant object for v(x, t)
        periodic:       list of 3 bools, periodic[i] is True if the flow is periodic in the ith coordinate. Time is i=3.
        bool_unsteady:  bool, specifies if velocity field is unsteady/steady
    
    Returns:

        y_update:       array (2, Npoints), updated position (=flow map) 
        y_prime_update: array (2, Npoints), updated velocity (=time derivative of flow map) 
    '''

    t0 = t
    
    # Compute x_prime at the beginning of the time-step by re-orienting and rescaling the vector field
    x_prime = velocity(t0, x1, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data)
    
    # compute derivative
    k1 = dt * x_prime

    # Update position at the first midpoint.
    x2 = x1 + .5 * k1
     
    # Update time
    t = t0+.5*dt
    
    # Compute x_prime at the first midpoint.
    x_prime = velocity(t, x2, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data)
    
    # compute derivative
    k2 = dt * x_prime

    # Update position at the second midpoint.
    x3 = x1 + .5 * k2
    
    # Update time
    t = t0+.5*dt
    
    # Compute x_prime at the second midpoint.
    x_prime = velocity(t, x3, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data)
    
    # compute derivative
    k3 = dt * x_prime
    
    # Update position at the endpoint.
    x4 = x1 + k3
    
    # Update time
    t = t0+dt
    
    # Compute derivative at the end of the time-step.
    x_prime = velocity(t, x4, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data) 
    
    # compute derivative
    k4 = dt * x_prime
    
    # Compute RK4 derivative
    y_prime_update = 1.0 / 6.0*(k1 + 2 * k2 + 2 * k3 + k4)
    
    # Integration y <-- y + y_prime*dt
    y_update = x1 + y_prime_update
    
    return y_update, y_prime_update/dt



