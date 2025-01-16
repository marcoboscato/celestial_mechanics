"""
==================================
Utilities to deal with Nbody units
==================================

This module contains functions and utilities to deal with N-body units
and units conversion

"""
from __future__ import annotations
from typing import Optional, Tuple, Callable, Union
import numpy as np
import numpy.typing as npt

class Nbody_units:
    """
    This class is used to handle transformation from and to Nbody Units.
    Internally it works considering the physical units in

        - mass: Msaturn
        - length: 151410 km (Epimetheus semi-major axis)
        - velocity: km/s
        - time: yr
    """

    Gcgs = 6.67430e-8 #Gravitational constants in cgs

    km_to_cm = 1e+5 #From km to cm

    Msaturn_cgs = 5.683e29 #from Msaturn to gr
    kg_to_gr = 1e3 #from kg to gr

    cms_to_kms = 1e-5 #cm/s to km/s
    s_to_yr = 3.1709791983765E-8 #s to yr


    def __init__(self, M: float = 1., L: float = 151410., V: float = 1., T: float = 1.):
        """
        Class initialised.
        The standard units are assumed in Msaturn (for the Mass), in km (for the length),
        in km/s (for the velocity), and in yr (for the time).
        These standards can be changes through the input parameter:

        :param M:  Set the physical input Mass scale in units of Msun
        :param L:  Set the physical input length scale in units of km
        :param V:  Set the physical input velocity scale in units of km/s
        :param T:  Set the physical input time scale in units of yr
        """
        cms_to_kms = 1e-5 #cm/s to km/s
        s_to_yr = 3.1709791983765E-8 #s to yr

        # This are the units that are used as in input in units of
        self.Lunits_scale = L  # km (Janus semi maj axis)
        self.Munits_scale = M  # Msaturn
        self.Vunits_scale = V  # km/s
        self.Tunits_scale = T  # yr

        Gscale_cgs = Nbody_units.Gcgs
        self.Lscale = L  # km (Epimetheus semi-major axis)
        self.Mscale = M  # Msaturn
        Mscale_cgs = self.Munits_scale * Nbody_units.Msaturn_cgs  # scale from Saturn Mass to gr
        Lscale_cgs = self.Lunits_scale * Nbody_units.km_to_cm      # scale from Epimetheus semi maj axis to gr (km to gr)
        self.Vscale = cms_to_kms * np.sqrt(Gscale_cgs * Mscale_cgs / Lscale_cgs)  # km/s
        self.Tscale = s_to_yr * np.sqrt(Lscale_cgs ** 3 / (Gscale_cgs * Mscale_cgs))  # yr


    ## 1) POSITION TRANSFORMATIONS
    def pos_to_Nbody(self, pos: Union[npt.NDArray[np.float64],float]) -> npt.NDArray[np.float64]:
        """
        Transform positions from physics to Nbody units.

        :param pos: A Nx2 numpy array containing the positions of the N particles, the physics units
                    should be consistent with the scale used at the class initialisation
        :return: A Nx2 numpy array containing the positions of the N particles scaled to Nbody units
        """

        # First transform input to km, then rescale by Lscale 
        return pos*self.Lunits_scale / self.Lscale

    def pos_to_km(self, pos: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Transform positions from  Nbody units to km

        :param pos:A Nx2 numpy array containing the positions of the N particles in Nbody units
        :return: A Nx2 numpy array containing the positions of the N particles in km
        """

        return pos * self.Lscale
    
    def pos_to_physical(self, pos: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Transform positions  Nbody units to  physical units, in this case in Epimetheus semi-major axis.

        :param pos:A Nx2 numpy array containing the positions of the N particles in Nbody units
        :return: A Nx2 numpy array containing the positions of the N particles in physical units
        """

        # First transform Nbody input  to km  ( pos*self.Lscae)
        # Then rescale by input units

        return self.pos_to_km(pos) / self.Lunits_scale

    ## 2) VELOCITY TRANSFORMATIONS
    def vel_to_Nbody(self, vel: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Transform velocities from physics to Nbody units

        :param vel: A Nx2 numpy array containing the velocity of the N particles. the physics units
                    should be consistent with the scale used at the class initialisation
        :return: A Nx2 numpy array containing the velocities of the N particles scaled to Nbody units
        """

        # First transform input to km/s then rescale by Vscale to get Nbody units
        return vel*self.Vunits_scale / self.Vscale

    def vel_to_kms(self, vel: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Transform velocities from  Nbody units to standard physics units km/s

        :param vel: A Nx2 numpy array containing the velocities of the N particles in Nbody units
        :return: A Nx2 numpy array containing the velocities of the N particles standard physics units (km/s)
        """

        return vel * self.Vscale

    def vel_to_physical(self, vel: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Transform velocities from Nbody  to physics units  (in km/s)
        The physics units should be consistent with the scale used at the class initialisation (e.g. V=1 units are in km/s,
        V=1e-3 units are in m/s).

        :param vel: A Nx2 numpy array containing the velocities of the N particles in Nbody units
        :return: A Nx2 numpy array containing the velocities of the N particles scaled in physics units
        """

        # First transform input to km/s Then transform to input units, IN THIS CASE IS EQUAL TO "vel_to kms()"
        return self.vel_to_kms(vel)/self.Vunits_scale

    ## 3) MASS TRANSFORMATIONS
    def m_to_Nbody(self, mass: Union[npt.NDArray[np.float64],float]) -> Union[npt.NDArray[np.float64],float] :
        """
        Transform masses from physics to Nbody units

        :param mass: A Nx1 numpy array (or a number) containing the mass of the  particles. the physics units
                    should be consistent with the scale used at the class initialisation
        :return: A Nx1 numpy array (or a number) containing the masses of the N particles scaled to Nbody units
        """

        return mass*self.Munits_scale / self.Mscale

    def m_to_Msaturn(self, mass: Union[npt.NDArray[np.float64],float] ) -> Union[npt.NDArray[np.float64],float] :
        """
        Transform ,asses from  Nbody units to standard physics units Msaturn

        :param mass: A Nx1 numpy array (or a number) containing the mass of the  particles in Nbody units
        :return: A Nx1 numpy array (or a number) containing the masses of the N particles in standard physics units (Msaturn)
        """

        return mass * self.Mscale

    def m_to_Kg(self, mass: Union[npt.NDArray[np.float64],float] ) -> Union[npt.NDArray[np.float64],float] :
        """
        Transform masses from Nbody units to physics, THIS CASE IN Kg

        :param mass: A Nx1 numpy array (or a number) containing the masses of the N particles in Nbody units.
        :return: A Nx1 numpy array (or a nunmber) containing the masses of the N particles scaled in physical units. The physics units
                    should be consistent with the scale used at the class initialisation (e.g. M=1 units are in Msun,
                    M=1e10 units are 10^{10} Msun).
        """
        return self.m_to_Msaturn(mass) * Nbody_units.Msaturn_cgs / Nbody_units.kg_to_gr  # from gr to Kg

    ## 4) TIME TRANFORMATIONS
    def t_to_Nbody(self, t: Union[npt.NDArray[np.float64],float]) -> Union[npt.NDArray[np.float64],float] :
        """
        Transform times from physics to Nbody units

        :param t: A Nx1 numpy array (or a number) containing time(s) in physics units. the physics units
                    should be consistent with the scale used at the class initialisation
        :return: A Nx1 numpy array (or a number) containing the time(s)  scaled to Nbody units
        """

        return t *self.Tunits_scale / self.Tscale

    def t_to_yr(self, t):
        """
        Transform times from Nbody units to standard physics units (yr)

        :param t: A Nx1 numpy array (or a number) containing time(s) in Nbody units.
        :return: A Nx1 numpy array (or a nunmber) containing the time(s) scaled in  standard physics units (yr).
        """

        return t * self.Tscale

    def t_to_days(self, t): # SAME RESULT AS t_to_yr
        """
        Transform time(s) from Nbody units to physics (in yr)

        :param t: A Nx1 numpy array (or a number) containing the time(s)  scaled to Nbody units
        :return: A Nx1 numpy array (or a number) containing time(s) in physics units. the physics units
                    should be consistent with the scale used at the class initialisation (e.g. T=1 units are in Myr,
                    T=1e3 units are Gyr).
        """

        return self.t_to_yr(t) * 365.25