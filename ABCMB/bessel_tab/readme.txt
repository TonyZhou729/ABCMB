This directory stores the precomputed spherical radial functions.
These are read and interpolated in order to compute CMB Cl's. 

File descriptions:
- l.txt : Integer values of l corresponding to columns in the radial function files. 
          The values of l were picked using CLASS' logstep->linearstep connvention with l_logstep=1.12, l_linstep=40
- x.txt : Float arguments where the radial functions were evaluated.
- phi0.txt : Radial function for the T0 Transfer Function.
- phi1.txt : Radial function for the T1 Transfer Function.
- phi2.txt : Radial function for the T2 Transfer Function.
- epsilon.txt : Radial function for the E-mode polarization Transfer Function.
- jl_stop.txt : The smallest x values of jl for l values in l.txt where the function exceeds 1.e5. The code stops integrating when x reaches these values.