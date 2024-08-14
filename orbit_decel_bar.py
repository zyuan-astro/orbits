#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 07:18:01 2022

@author: cdli
"""


import agama, numpy, sys, os
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.coordinates as coord
import astropy.units as u
try:
    from ConfigParser import RawConfigParser  # python 2
except ImportError:
    from configparser import RawConfigParser  # python 3
v_sun = coord.CartesianDifferential([11.1, 12.24+238 , 7.25]*u.km/u.s)  #velocity of the sun, V = 12.24 + 238
gc_frame = coord.Galactocentric(galcen_distance=8.2*u.kpc,
                                        galcen_v_sun=v_sun,
                                        z_sun=20*u.pc)

# write out the rotation curve (separately for each component, and the total one)
def writeRotationCurve(filename, potentials):
    radii = numpy.logspace(-2., 2., 81)
    xyz   = numpy.column_stack((radii, radii*0, radii*0))
    vcomp = numpy.column_stack([(-potential.force(xyz)[:,0] * radii)**0.5 for potential in potentials])
    vtot  = numpy.sum(vcomp**2, axis=1)**0.5
    numpy.savetxt(filename, numpy.column_stack((radii, vtot, vcomp)), fmt="%.6g", delimiter="\t", \
        header="radius[Kpc]\tv_circ,total[km/s]\tdisk\tbulge\thalo")

# print surface density profiles to a file
def writeSurfaceDensityProfile(filename, model):
    print("Writing surface density profile")
    radii = numpy.hstack(([1./8, 1./4], numpy.linspace(0.5, 16, 32), numpy.linspace(18, 30, 7)))
    xy    = numpy.column_stack((radii, radii*0))
    Sigma = model.moments(xy, dens=True, vel=False, vel2=False, separate=True) * 1e-6  # convert from Msun/Kpc^2 to Msun/pc^2
    numpy.savetxt(filename, numpy.column_stack((radii, Sigma)), fmt="%.6g", delimiter="\t", \
        header="Radius[Kpc]\tsurfaceDensity[Msun/pc^2]")

# print vertical density profile for several sub-components of the stellar DF
def writeVerticalDensityProfile(filename, model):
    print("Writing vertical density profile")
    height = numpy.hstack((numpy.linspace(0, 1.5, 13), numpy.linspace(2, 8, 13)))
    xyz   = numpy.column_stack((height*0 + solarRadius, height*0, height))
    dens  = model.moments(xyz, dens=True, vel=False, vel2=False, separate=True) * 1e-9  # convert from Msun/Kpc^3 to Msun/pc^3
    numpy.savetxt(filename, numpy.column_stack((height, dens)), fmt="%.6g", delimiter="\t", \
        header="z[Kpc]\tThinDisk\tThickDisk\tStellarHalo[Msun/pc^3]")

# print velocity distributions at the given point to a file
def writeVelocityDistributions(filename, model):
    point = (solarRadius, 0, 0.1)
    print("Writing velocity distributions at (R=%g, z=%g)" % (point[0], point[2]))
    # create grids in velocity space for computing the spline representation of VDF
    v_max = 360.0    # km/s
    gridv = numpy.linspace(-v_max, v_max, 75) # use the same grid for all dimensions
    # compute the distributions (represented as cubic splines)
    splvx, splvy, splvz = model.vdf(point, gridv)
    # output f(v) at a different grid of velocity values
    gridv = numpy.linspace(-v_max, v_max, 201)
    numpy.savetxt(filename, numpy.column_stack((gridv, splvx(gridv), splvy(gridv), splvz(gridv))),
        fmt="%.6g", delimiter="\t", header="V\tf(V_x)\tf(V_y)\tf(V_z) [1/(km/s)]")

# display some information after each iteration
def printoutInfo(model, iteration):
    densDisk = model.components[0].getDensity()
    densBulge= model.components[1].getDensity()
    densHalo = model.components[2].getDensity()
    pt0 = (solarRadius, 0, 0)
    pt1 = (solarRadius, 0, 1)
    print("Disk total mass=%g Msun, rho(Rsolar,z=0)=%g, rho(Rsolar,z=1kpc)=%g Msun/pc^3" % \
        (densDisk.totalMass(), densDisk.density(pt0)*1e-9, densDisk.density(pt1)*1e-9))  # per pc^3, not kpc^3
    print("Halo total mass=%g Msun, rho(Rsolar,z=0)=%g, rho(Rsolar,z=1kpc)=%g Msun/pc^3" % \
        (densHalo.totalMass(), densHalo.density(pt0)*1e-9, densHalo.density(pt1)*1e-9))
    print("Potential at origin=-(%g km/s)^2, total mass=%g Msun" % \
        ((-model.potential.potential(0,0,0))**0.5, model.potential.totalMass()))
    densDisk.export ("dens_disk_" +iteration);
    densBulge.export("dens_bulge_"+iteration);
    densHalo.export ("dens_halo_" +iteration);
    model.potential.export("potential_"+iteration);
    writeRotationCurve("rotcurve_"+iteration, (model.potential[1],  # disk potential (CylSpline)
        agama.Potential(type='Multipole', lmax=6, density=densBulge),        # -"- bulge
        agama.Potential(type='Multipole', lmax=6, density=densHalo) ) )      # -"- halo


if __name__ == "__main__":
    # read parameters from the INI file
    iniFileName = "/place/to/galaxymodel/SCM.ini"
    ini = RawConfigParser()
    ini.optionxform=str  # do not convert key to lowercase
    ini.read(iniFileName)
    iniPotenThinDisk = dict(ini.items("Potential thin disk"))
    iniPotenThickDisk= dict(ini.items("Potential thick disk"))
    iniPotenGasDisk  = dict(ini.items("Potential gas disk"))
    iniPotenBulge    = dict(ini.items("Potential bulge"))
    iniPotenDarkHalo = dict(ini.items("Potential dark halo"))
    iniDFThinDisk    = dict(ini.items("DF thin disk"))
    iniDFThickDisk   = dict(ini.items("DF thick disk"))
    iniDFStellarHalo = dict(ini.items("DF stellar halo"))
    iniDFDarkHalo    = dict(ini.items("DF dark halo"))
    iniDFBulge       = dict(ini.items("DF bulge"))
    iniSCMHalo       = dict(ini.items("SelfConsistentModel halo"))
    iniSCMBulge      = dict(ini.items("SelfConsistentModel bulge"))
    iniSCMDisk       = dict(ini.items("SelfConsistentModel disk"))
    iniSCM           = dict(ini.items("SelfConsistentModel"))
    solarRadius      = ini.getfloat("Data", "SolarRadius")

    # define external unit system describing the data (including the parameters in INI file)
    agama.setUnits(length=1, velocity=1, mass=1)   # in Kpc, km/s, Msun

    # initialize the SelfConsistentModel object (only the potential expansion parameters)
    model = agama.SelfConsistentModel(**iniSCM)

    # create initial ('guessed') density profiles of all components
    densityBulge       = agama.Density(**iniPotenBulge)
    densityDarkHalo    = agama.Density(**iniPotenDarkHalo)
    densityThinDisk    = agama.Density(**iniPotenThinDisk)
    densityThickDisk   = agama.Density(**iniPotenThickDisk)
    densityGasDisk     = agama.Density(**iniPotenGasDisk)
    densityStellarDisk = agama.Density(densityThinDisk, densityThickDisk)  # composite

    # add components to SCM - at first, all of them are static density profiles
    model.components.append(agama.Component(density=densityStellarDisk, disklike=True))
    model.components.append(agama.Component(density=densityBulge,       disklike=False))
    model.components.append(agama.Component(density=densityDarkHalo,    disklike=False))
    model.components.append(agama.Component(density=densityGasDisk,     disklike=True))

    # compute the initial potential
    model.iterate()
    printoutInfo(model, "init")

    print("\033[1;33m**** STARTING MODELLING ****\033[0m\nInitial masses of density components: " \
        "Mdisk=%g Msun, Mbulge=%g Msun, Mhalo=%g Msun, Mgas=%g Msun" % \
        (densityStellarDisk.totalMass(), densityBulge.totalMass(), \
        densityDarkHalo.totalMass(), densityGasDisk.totalMass()))

    # create the dark halo DF
    dfHalo  = agama.DistributionFunction(potential=model.potential, **iniDFDarkHalo)
    # same for the bulge
    dfBulge = agama.DistributionFunction(potential=model.potential, **iniDFBulge)
    # same for the stellar components (thin/thick disks and stellar halo)
    dfThinDisk    = agama.DistributionFunction(potential=model.potential, **iniDFThinDisk)
    dfThickDisk   = agama.DistributionFunction(potential=model.potential, **iniDFThickDisk)
    dfStellarHalo = agama.DistributionFunction(potential=model.potential, **iniDFStellarHalo)
    # composite DF of all stellar components except the bulge
    dfStellar     = agama.DistributionFunction(dfThinDisk, dfThickDisk, dfStellarHalo)
    # composite DF of all stellar components including the bulge
    dfStellarAll  = agama.DistributionFunction(dfThinDisk, dfThickDisk, dfStellarHalo, dfBulge)

    # replace the disk, halo and bulge SCM components with the DF-based ones
    model.components[0] = agama.Component(df=dfStellar, disklike=True, **iniSCMDisk)
    model.components[1] = agama.Component(df=dfBulge, disklike=False, **iniSCMBulge)
    model.components[2] = agama.Component(df=dfHalo,  disklike=False, **iniSCMHalo)

    # we can compute the masses even though we don't know the density profile yet
    print("Masses of DF components: " \
        "Mdisk=%g Msun (Mthin=%g, Mthick=%g, Mstel.halo=%g); Mbulge=%g Msun; Mdarkhalo=%g Msun" % \
        (dfStellar.totalMass(), dfThinDisk.totalMass(), dfThickDisk.totalMass(), \
        dfStellarHalo.totalMass(), dfBulge.totalMass(), dfHalo.totalMass()))

    # do a few more iterations to obtain the self-consistent density profile for the entire system
    for iteration in range(1,4):
        print("\033[1;37mStarting iteration #%d\033[0m" % iteration)
        model.iterate()
        printoutInfo(model, "iter"+str(iteration))

# Modification of equation 9 of Coleman et al. 2020 (https://arxiv.org/abs/1911.04714)
def makeXBar(**params):
    densityNorm = params['densityNorm']
    x0   = params['x0']
    y0   = params['y0']
    z0   = params['z0']
    xc   = params['xc']
    yc   = params['yc']
    c    = params['c']
    alpha= params['alpha']
    cpar = params['cpar']
    cperp= params['cperp']
    m    = params['m']
    n    = params['n']
    outerCutoffRadius = params['outerCutoffRadius']
    def density(xyz):
        r  = numpy.sum(xyz**2, axis=1)**0.5
        a  = ( ( (abs(xyz[:,0]) / x0)**cperp + (abs(xyz[:,1]) / y0)**cperp )**(cpar/cperp) +
            (abs(xyz[:,2]) / z0)**cpar )**(1/cpar)
        ap = ( ((xyz[:,0] + c * xyz[:,2]) / xc)**2 + (xyz[:,1] / yc)**2 )**(0.5)
        am = ( ((xyz[:,0] - c * xyz[:,2]) / xc)**2 + (xyz[:,1] / yc)**2 )**(0.5)
        return (densityNorm / numpy.cosh(a**m) * numpy.exp( -(r/outerCutoffRadius)**2) *
            (1 + alpha * (numpy.exp(-ap**n) + numpy.exp(-am**n) ) ) )
    return density

# Modification of equation 9 of Wegg et al. 2015 (https://arxiv.org/pdf/1504.01401.pdf)
def makeLongBar(**params):
    densityNorm = params['densityNorm']
    x0   = params['x0']
    y0   = params['y0']
    cpar = params['cpar']
    cperp= params['cperp']
    scaleHeight = params['scaleHeight']
    innerCutoffRadius   = params['innerCutoffRadius']
    outerCutoffRadius   = params['outerCutoffRadius']
    innerCutoffStrength = params['innerCutoffStrength']
    outerCutoffStrength = params['outerCutoffStrength']
    def density(xyz):
        R = (xyz[:,0]**2 + xyz[:,1]**2)**0.5
        a = ( (abs(xyz[:,0]) / x0)**cperp + (abs(xyz[:,1]) / y0)**cperp )**(1/cperp)
        return densityNorm / numpy.cosh(xyz[:,2] / scaleHeight)**2 * numpy.exp(-a**cpar
            -(R/outerCutoffRadius)**outerCutoffStrength - (innerCutoffRadius/R)**innerCutoffStrength)
    return density

def makeDensityModel():
    params = numpy.array(
    # short/thick bar
    # [ 3.16273226e+09, 4.90209137e-01, 3.92017253e-01, 2.29482096e-01,
    #   1.99110223e+00, 2.23179266e+00, 8.73227940e-01, 4.36983774e+00,
    #   6.25670015e-01, 1.34152138e+00, 1.94025114e+00, 7.50504078e-01,
    #   4.68875471e-01] +
    # long bar 1
    [ 4.95381575e+08, 5.36363324e+00, 9.58522229e-01, 6.10542494e-01,
      9.69645220e-01, 3.05125124e+00, 3.19043585e+00, 5.58255674e-01,
      1.67310332e+01, 3.19575493e+00] +
    # long bar 2
    [ 1.74304936e+13, 4.77961423e-01, 2.66853061e-01, 2.51516920e-01,
      1.87882599e+00, 9.80136710e-01, 2.20415408e+00, 7.60708626e+00,
      -2.72907665e+01, 1.62966434e+00]
    )
    ind=0
    # densityXBar = makeXBar(
    #      densityNorm=params[ind+0],
    #      x0=params[ind+1],
    #      y0=params[ind+2],
    #      z0=params[ind+3],
    #      cpar=params[ind+4],
    #      cperp=params[ind+5],
    #      m=params[ind+6],
    #      outerCutoffRadius=params[ind+7],
    #      alpha=params[ind+8],
    #      c=params[ind+9],
    #      n=params[ind+10],
    #      xc=params[ind+11],
    #      yc=params[ind+12])
    # ind+=13
    densityLongBar1 = makeLongBar(
        densityNorm=params[ind+0],
        x0=params[ind+1],
        y0=params[ind+2],
        scaleHeight=params[ind+3],
        cperp=params[ind+4],
        cpar=params[ind+5],
        outerCutoffRadius=params[ind+6],
        innerCutoffRadius=params[ind+7],
        outerCutoffStrength=params[ind+8],
        innerCutoffStrength=params[ind+9] )
    ind+=10
    densityLongBar2 = makeLongBar(
        densityNorm=params[ind+0],
        x0=params[ind+1],
        y0=params[ind+2],
        scaleHeight=params[ind+3],
        cperp=params[ind+4],
        cpar=params[ind+5],
        outerCutoffRadius=params[ind+6],
        innerCutoffRadius=params[ind+7],
        outerCutoffStrength=params[ind+8],
        innerCutoffStrength=params[ind+9] )
    ind+=10
    assert len(params)==ind, 'invalid number of parameters'
    return agama.Density(lambda x: densityLongBar1(x) + densityLongBar2(x), symmetry='t')


# create the potential of the entire model:
# 4-component stellar density as defined above, plus central mass concentration, plus dark halo
def makebarPotential():
    # combined 4 components and the CMC represented by a single triaxial CylSpline potential
    mmax = 12  # order of azimuthal Fourier expansion (higher order means better accuracy,
    # but values greater than 12 *significantly* slow down the computation!)
    pot_bar = agama.Potential(type='CylSpline',
        density=agama.Density(makeDensityModel()),
        symmetry='t', mmax=mmax, gridsizeR=25, gridsizez=25, Rmin=0.1, Rmax=40, zmin=0.05, zmax=20)
    return agama.Potential(pot_bar)


den = makeDensityModel()
pot_bar = makebarPotential()
pot_back = model.potential 
pot = agama.Potential(pot_bar, pot_back[1], pot_back[0]) 

rot_angle = numpy.array([
        [ -6,             80+328],
        [-3,     -160+328],
        [ 0,             -300+328],
        ])
rot_angle_spl = agama.CubicSpline(rot_angle[:,0], rot_angle[:,1])
pattern_slow = rot_angle_spl(rot_angle[:,0], der=1)
print('Pattern speed is %.3g, %.3g, %.3g' %(pattern_slow[0],pattern_slow[1],pattern_slow[2]))

numpy.savetxt('/place/to/decelbar_sp/rot_angle.txt', rot_angle)

M_f = 2.0  # multiplier for the bar mass at the final time t_f
R_f = 1.2  # multiplier for the radius at t_f

# first approach: create extra points with duplicate values
bar_scale1 = numpy.array([
        [-6, 1.0, 1.0],
            [0, M_f, R_f]])
mass_scale_spl1 = agama.CubicSpline(bar_scale1[:,0], bar_scale1[:,1], reg=True)  # regularized spline,
size_scale_spl1 = agama.CubicSpline(bar_scale1[:,0], bar_scale1[:,2], reg=True)

# print(size_scale_spl1([-6,-3,0], der=0))
# print(mass_scale_spl1([-6,-3,0], der=0))


numpy.savetxt('/place/to/decelbar_sp/bar_scale.txt', bar_scale1)
pot_bar_slowing_growing = agama.Potential(potential=pot_bar, 
    scale='/place/to/decelbar_sp/bar_scale.txt', rotation='/place/to/decelbar_sp/rot_angle.txt')



def createSpiralPotential(numberOfArms, surfaceDensity, scaleRadius, scaleHeight, pitchAngle, phi0=0):
    '''
    Create a CylSpline approximation for a spiral arm potential from Cox&Gomez(2002).
    The density generated by this potential approximately follows an exponential
    disk profile with the amplitude of surface density variation
    Sigma(R) = surfaceDensity * exp( - R / scaleRadius ),
    and isothermal vertical profile
    rho(R,z) = Sigma(R) / (4h) * sech^2( z / (2h) ),
    where h is the scaleHeight (note the extra factor 1/2 w.r.t. the original paper,
    which is introduced to match the definition of the Disk profile in Agama).
    The density at the crest of the spiral arm is  ~ 3 * rho(R,z=0),
    while in the inter-arm region it is ~ (-1.5 to -1) * rho(R,z=0).
    The spiral is logarithmic, so that the azimuthal angle (of one of the arms) is
    phi(R) ~ phi0 + ln(R/scaleRadius) / tan(pitchAngle).
    '''
    def potfnc(xyz):
        R = (xyz[:,0]**2 + xyz[:,1]**2)**0.5
        z = xyz[:,2]
        phi = numpy.arctan2(xyz[:,1], xyz[:,0])
        prefac = -4*numpy.pi * agama.G * surfaceDensity * numpy.exp(-R / scaleRadius)
        gamma  = numberOfArms * (phi - numpy.log(R / scaleRadius) / numpy.tan(pitchAngle) - phi0)
        Phi = numpy.zeros(len(R))
        for n in range(1,4):
            K_n   = n * numberOfArms / R / numpy.sin(pitchAngle)
            K_n_H = K_n * 2 * scaleHeight
            B_n   = K_n_H * (1 + 0.4 * K_n_H)
            D_n   = 1 / (1 + 0.3 * K_n_H) + K_n_H
            C_n   = [8./3/numpy.pi, 0.5, 8./15/numpy.pi][n-1]  # amplitudes of the three harmonic terms
            Phi  += prefac * ( C_n / D_n / K_n * numpy.cos(n * gamma) *
                (numpy.cosh(K_n * z / B_n))**-B_n )
        return numpy.nan_to_num(Phi)  # VERY IMPORTANT is to make sure it never produces a NaN

    # now create a CylSpline potential approximating this user-defined profile,
    # paying special attention to the grid parameters
    return agama.Potential(type='CylSpline', potential=potfnc,
    Rmin = 0.01 * scaleRadius,  # (these could be adjusted if needed)
    Rmax = 10.0 * scaleRadius,
    zmin = 0.25 * scaleHeight,
    zmax = 10.0 * scaleHeight,
    mmax = 3 * numberOfArms,    # each arm is represented by three harmonic terms: m, m*2, m*3
    symmetry  = 'bisymmetric' if numberOfArms % 2 == 0 else 4,  # 4 means the z-reflection symmetry
    gridSizeZ = 20,
    # rule of thumb for the radial grid spacing is to resolve
    # the change in the pitch angle of the spiral with one grid cell
    gridSizeR = max(25, numpy.log(10.0 / 0.01) / numpy.tan(pitchAngle) * numberOfArms))

numberOfArms    = 2
pitchAngle      = numpy.radians(9.9)
scaleRadius     = 1
scaleHeight     = 0.1
surfaceDensityS = 2.5*10**9  # density of the underlying axisymmetric disk
#surfaceDensityS = surfaceDensityD * 0.5  # amplitude of the density variation of the spiral pattern
phi0            = numpy.radians(18.9*.35+26)   # [arbitrary] phase angle at R=scaleRadius
pot_spiral = createSpiralPotential(numberOfArms, surfaceDensityS, scaleRadius, scaleHeight, pitchAngle, phi0)


#Omega_sp = -18.9
  # time at which the potential starts to slow down, again in our time units
  # time at which it achieves the final pattern speed
rot_angle_sp = numpy.array([
    [ -6,            139.4],
    [ 0,             26],
])
rot_angle_spl_sp = agama.CubicSpline(rot_angle_sp[:,0], rot_angle_sp[:,1])
pattern_slow_sp = rot_angle_spl_sp(rot_angle_sp[:,0], der=1)
print('Pattern speed of spiral is %.3g, %.3g' %(pattern_slow_sp[0],pattern_slow_sp[1]))

numpy.savetxt('/place/to/decelbar_sp/rot_angle_sp.txt', rot_angle_sp)


pot_spiral_rot = agama.Potential(potential=pot_spiral, 
                                 rotation='/place/to/decelbar_sp/rot_angle_sp.txt')


pot_mw_slowing = agama.Potential(pot_spiral_rot, pot_bar_slowing_growing, pot_back[1], pot_back[0]) 

sample_points = numpy.load('/place/to/vmp_sample.npy', allow_pickle=True)

c = sample_points.reshape(284*500,7)

icrs_star = coord.ICRS(ra=c[:,1]*u.degree, dec=c[:,2]*u.degree,distance=c[:,5]*u.kpc,pm_ra_cosdec=c[:,3]
                                       *u.mas/u.yr,pm_dec=c[:,4]*u.mas/u.yr,radial_velocity=c[:,6]*u.km/u.s)
gc_star = icrs_star.transform_to(gc_frame)

xo  = numpy.array(gc_star.x)
yo  = numpy.array(gc_star.y)
zo  = numpy.array(gc_star.z)
vxo = numpy.array(gc_star.v_x)
vyo = numpy.array(gc_star.v_y)
vzo = numpy.array(gc_star.v_z)


ic1 = numpy.stack((xo, yo, zo, vxo, vyo, vzo), axis=-1)

orbits1 = agama.orbit(potential=pot_mw_slowing, ic=ic1, time=-6., trajsize=301)#[:,1]
orbit_final = numpy.stack((orbits1[:,1],c[:,0]), axis=-1)
print(orbits1[:,0][0])
#----------------------save the orbits to file---------------------------------------------
numpy.save('/place/to/decelbar_sp/orbits',orbit_final)

