from typing import Optional, Tuple, Callable, Union, List
import numpy as np
import numpy.typing as npt
from Particle import Particles

def integrator_rungekutta(particles: Particles,
                        tstep: float,
                        acceleration_estimator: Union[Callable,List]):

    mass = particles.mass

    particles.acc = acceleration_estimator(Particles(particles.pos ,
                                                     particles.vel ,
                                                     particles.mass))
    
    k1_r = tstep * particles.vel
    k1_v = tstep * particles.acc

    k2_r = tstep * (particles.vel + 0.5 * k1_v)
    k2_v = tstep * acceleration_estimator(Particles(particles.pos + 0.5 * k1_r,
                                                    particles.vel + 0.5 * k1_v, 
                                                    mass))

    k3_r = tstep * (particles.vel + 0.5 * k2_v)
    k3_v = tstep * acceleration_estimator(Particles(particles.pos + 0.5 * k2_r,
                                                    particles.vel + 0.5 * k2_v,
                                                    mass))

    k4_r = tstep * (particles.vel + k3_v)
    k4_v = tstep * acceleration_estimator(Particles(particles.pos + k3_r,
                                                    particles.vel + k3_v,
                                                    mass))

    particles.pos = particles.pos + (k1_r + 2 * k2_r + 2 * k3_r + k4_r) / 6
    particles.vel = particles.vel + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6


    # Now return the updated particles, the acceleration, jerk (can be None) and potential (can be None)

    return particles