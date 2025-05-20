from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt
from .moons import Moons

try:
    import pyfalcon
    pyfalcon_load=True
except:
    pyfalcon_load=False


def acceleration_pyfalcon(particles: Moons, softening: float =0.) \
        -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
    """
    Estimate the acceleration following the fast-multipole gravity Dehnen2002 solver (https://arxiv.org/pdf/astro-ph/0202512.pdf)
    as implementd in pyfalcon (https://github.com/GalacticDynamics-Oxford/pyfalcon)

    :param particles: An instance of the class Particles
    :param softening: Softening parameter
    :return: A tuple with 3 elements:

        - Acceleration: a NX3 numpy array containing the acceleration for each particle
        - Jerk: None, the jerk is not estimated
        - Pot: a Nx1 numpy array containing the gravitational potential at each particle position
    """

    if not pyfalcon_load: return ImportError("Pyfalcon is not available")

    acc, pot = pyfalcon.gravity(particles.pos,particles.mass,softening,kernel=0)
    jerk = None

    return acc, jerk, pot



def acceleration_direct(particles: Moons, softening: float =0.) \
    -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:

    def Force(mass, acc):
        return mass*acc

#    N           = 100
#    pos_min     = 5.0
#    pos_max     = 10.0
#    vel_min     = 1.0
#    vel_max     = 10.0
#    mass_min    = 1.0
#    mass_max    = 100.0

    pos     = particles.pos
    v       = particles.vel
    mass    = particles.mass 
    N       = len(particles)

    acc     = np.zeros((N,2),float)
    force   = np.zeros((N,2),float)

    for i in range(N):
        for j in range(N):
            if i != j:
                denom      = np.linalg.norm(pos[i] - pos[j])**3
                #denom      = (np.sum((pos[i] - pos[j])**2.) + softening**2.)**(1.5)
                temp       = - mass[j]*(pos[i] - pos[j]) / denom
                acc[i,:]   = acc[i,:] + temp

        force[i,:] = Force(mass[i], acc[i,:])
 
    jerk = None
    pot = None

    return (acc,jerk,pot)


def acceleration_direct_vectorized(particles: Moons, softening: float =0.) \
    -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:

    pos     = particles.pos # particles'positions
    v       = particles.vel # particles'velocities
    mass    = particles.mass # particles'masses
    N       = len(particles)

    #x
    ax = pos[:,0]              # I concentrate on x
    bx = ax.reshape((N,1))     # I need to reshape to transform it in a vector (vertical)
    cx = bx - ax               # this is xi-xj, the delta. I am creating a matrix xi-xj (NxN)
    #y
    ay = pos[:,1]              #same for y
    by = ay.reshape((N,1))
    cy = by - ay

    r = np.array((cx, cy)) #I put everything into a sole tenson (3,N,N)
    
    deltax2 = r[0,:,:]**2
    deltay2 = r[1,:,:]**2

    normr  = np.sqrt(deltax2 + deltay2 + softening**2.)    #I calculate |r| and |r|**3
    normr3 = normr**3

    factor = r / normr3        # I calculate the factor in the expression for a; I still have a tensor (3,N,N)
    addend = mass*factor       # I construct my addend multiplying by the mass
    addend[np.isnan(addend)] = 0    #I substitute the nans in the diagonal with zeros

    addendx = addend[0,:,:]    #I devide in the three components
    addendy = addend[1,:,:]

    ax = - addendx.sum(axis=1)   #and I sum axis by axis
    ay = - addendy.sum(axis=1)

    acc = np.array((ax, ay)) #this is the acceleration suffered by each particle by each other in the three components matrix(N,3)
    acc = acc.T                    # To get a matrix (3,N)
    
    jerk = None
    pot = None

    return (acc,jerk,pot)