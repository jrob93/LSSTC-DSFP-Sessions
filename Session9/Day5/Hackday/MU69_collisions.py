'''Collision outcomes for MU69 components'''

import numpy
import py_func as pf

R1=6.9e3
R2=6.3e3
V_coll=5

V1=1400e9
V2=1050e9
R_sys=(3.0*(V1+V2)/(4.*numpy.pi))**(1.0/3.0)
print R_sys

# rho_vals=[1e2,5e2,1e3,1.5e3,2e3,2.5e3,3e3] # full range of densities
rho_vals=[5e2,1e3] # typical comet density to our chosen dnesity of water
# rho=1e3

for rho in rho_vals:

    print "\nrho = {} kg/m^3".format(rho)

    M1=(4.0/3.0)*numpy.pi*rho*(R1**3.0)
    M2=(4.0/3.0)*numpy.pi*rho*(R2**3.0)
    M_sys=M1+M2
    # R_sys=(3.0*M_sys/(4.0*numpy.pi*rho))**(1.0/3.0)

    v_esc=(2.0*pf.G*M_sys/R_sys)**(1.0/2.0)

    print "M1 = {} kg, M2 = {} kg, M_sys = {} kg".format(M1,M2,M_sys)
    print "R_sys = {} m".format(R_sys)
    print "v_esc = {} m/s".format(v_esc)
