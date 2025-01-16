"""
=========================================================
ODE integrators  (:mod:`fireworks.nbodylib.integrators`)
=========================================================

This module contains a collection of integrators to integrate one step of the ODE N-body problem
"""
from typing import Optional, Tuple, Callable, Union, List
import numpy as np
import numpy.typing as npt
from .moons import Moons


''' LEAPFROG '''

def integrator_leapfrog(particles: Moons,
                        tstep: float,
                        acceleration_estimator: Union[Callable,List],
                        softening: float = 0.,
                        external_accelerations: Optional[List] = None):

    acc,jerk,potential=acceleration_estimator(particles,softening)

    #Check additional accelerations
    if external_accelerations is not None:
        for ext_acc_estimator in external_accelerations:
            acct,jerkt,potentialt=ext_acc_estimator(particles,softening)
            acc+=acct
            if jerk is not None and jerkt is not None: jerk+=jerkt
            if potential is not None and potentialt is not None: potential+=potentialt

    # Velocity Verlet estimate
    #particles.set_acc(acc) # set acceleration
    particles.acc = acceleration_estimator(Moons(particles.pos ,
                                        particles.vel ,
                                        particles.mass ), softening)[0]
    particles.pos = particles.pos + tstep*particles.vel +(tstep**2./2.)*particles.acc
    particles.acc_old = np.copy(particles.acc)
    particles.acc = acceleration_estimator(Moons(particles.pos ,
                                                    particles.vel ,
                                                    particles.mass ), softening)[0]
    particles.vel = particles.vel + tstep/2.*(particles.acc + particles.acc_old)
    
    # Now return the updated particles, the acceleration, jerk (can be None) and potential (can be None)
    return (particles, tstep, acc, jerk, potential)



''' RUNGE-KUTTA '''

def integrator_rungekutta(particles: Moons,
                        tstep: float,
                        acceleration_estimator: Union[Callable,List],
                        softening: float = 0.,
                        external_accelerations: Optional[List] = None):

    acc,jerk,potential=acceleration_estimator(particles,softening)

    #Check additional accelerations
    if external_accelerations is not None:
        for ext_acc_estimator in external_accelerations:
            acct,jerkt,potentialt=ext_acc_estimator(particles,softening)
            acc+=acct
            if jerk is not None and jerkt is not None: jerk+=jerkt
            if potential is not None and potentialt is not None: potential+=potentialt

    mass = particles.mass

    particles.acc = acceleration_estimator(Moons(particles.pos ,
                                                     particles.vel ,
                                                     particles.mass ), softening)[0]
    
    k1_r = tstep * particles.vel
    k1_v = tstep * particles.acc

    k2_r = tstep * (particles.vel + 0.5 * k1_v)
    k2_v = tstep * acceleration_estimator(Moons(particles.pos + 0.5 * k1_r,
                                                    particles.vel + 0.5 * k1_v, 
                                                    mass), softening)[0]

    k3_r = tstep * (particles.vel + 0.5 * k2_v)
    k3_v = tstep * acceleration_estimator(Moons(particles.pos + 0.5 * k2_r,
                                                    particles.vel + 0.5 * k2_v,
                                                    mass), softening)[0]

    k4_r = tstep * (particles.vel + k3_v)
    k4_v = tstep * acceleration_estimator(Moons(particles.pos + k3_r,
                                                    particles.vel + k3_v,
                                                    mass), softening)[0]

    particles.pos = particles.pos + (k1_r + 2 * k2_r + 2 * k3_r + k4_r) / 6
    particles.vel = particles.vel + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6

    # Now return the updated particles, the acceleration, jerk (can be None) and potential (can be None)
    return (particles, tstep, acc, jerk, potential)
