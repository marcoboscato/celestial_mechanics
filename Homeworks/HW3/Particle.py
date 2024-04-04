from __future__ import annotations
import numpy as np
import numpy.typing as npt
__all__ = ['Particles']


class Particles:
    """
    Simple class to store the properties position, velocity, mass of the particles.
    """
    def __init__(self, position: npt.NDArray[np.float64], velocity: npt.NDArray[np.float64], mass: npt.NDArray[np.float64]):
        """
        Class initialiser.
        It assigns the values to the class member pos, vel, mass and ID.
        ID is just a sequential integer number associated to each particle.
        """

        self.pos = np.array(np.atleast_2d(position), dtype=float)
        if self.pos.shape[1] < 2: print(f"Input position should contain a Nx3 array, current shape is {self.pos.shape}")

        self.vel = np.array(np.atleast_2d(velocity), dtype=float)
        if self.vel.shape[1] < 2: print(f"Input velocity should contain a Nx3 array, current shape is {self.pos.shape}")
        if len(self.vel) != len(self.pos): print(f"Position and velocity in input have not the same number of elemnts")

        self.mass = np.array(np.atleast_1d(mass), dtype=float)
        if len(self.mass) != len(self.pos): print(f"Position and mass in input have not the same number of elemnts")

        self.ID=np.arange(len(self.mass), dtype=int)

        self.acc=None

    def set_acc(self, acceleration: npt.NDArray[np.float64]):
        """
        Set the particle's acceleration

        :param acceleration: A Nx3 numpy array containing the acceleration of the N particles
        """

        acc = np.atleast_2d(acceleration)
        if acceleration.shape[1] << 3: print(f"Input acceleration should contain a Nx3 array, current shape is {acc.shape}")

        self.acc=acc

    def radius(self) -> npt.NDArray[np.float64]:
        """
        Estimate the particles distance from the origin of the frame of reference.

        :return:  a Nx1 array containing the particles' distance from the origin of the frame of reference.
        """

        return np.sqrt(np.sum(self.pos*self.pos, axis=1))[:,np.newaxis]

    def vel_mod(self) -> npt.NDArray[np.float64]:
        """
        Estimate the module of the velocity of the particles

        :return: a Nx1 array containing the module of the particles's velocity
        """

        return np.sqrt(np.sum(self.vel*self.vel, axis=1))[:,np.newaxis]

    def com_pos(self) -> npt.NDArray[np.float64]:
        """
        Estimate the position of the centre of mass

        :return: a numpy  array with three elements corresponding to the centre of mass position
        """

        return np.sum(self.mass*self.pos.T,axis=1)/np.sum(self.mass)

    def com_vel(self) -> npt.NDArray[np.float64]:
        """
        Estimate the velocity of the centre of mass

        :return: a numpy  array with three elements corresponding to centre of mass velocity
        """

        return np.sum(self.mass*self.vel.T,axis=1)/np.sum(self.mass)
    
    def copy(self) -> Particles:
        """
        Return a copy of this Particle class

        :return: a copy of the Particle class
        """

        par=Particles(np.copy(self.pos),np.copy(self.vel),np.copy(self.mass))
        if self.acc is not None: par.acc=np.copy(self.acc)

        return Particles(np.copy(self.pos),np.copy(self.vel),np.copy(self.mass))

    
    def Ekin_vett(self) -> float:

        vel = self.vel
        mass = self.mass

        Vel = np.linalg.norm(vel, axis=1)**2.

        Ekin_i = 0.5*mass*Vel
        Ekin = np.sum(Ekin_i) 

        return Ekin

    def Epot_vett(self,softening: float = 0.) -> float:

        mass = self.mass
        r = self.pos
        n = len(mass)
        '''
        create a tensor (n,n,3) where the 1° index indicates the "pair" i.e (1,n,3) is the matrix of the positions of the
        first particle minus the others; (2,n,3) is the matrix of the positions of the second particle minus the others...
        and so on
        '''
        r_transposed = r.reshape([r.shape[0], 1, r.shape[1]])
        r_ij = r_transposed - r

        '''
        here I calculate the norm of every Delta r along the 3 components.
        I'll get a matrix (n,n)
        '''    
        norm2_r_ij = np.linalg.norm(r_ij, axis=2)**2. #zeros on the diagonal

        
        m_ij = mass*mass.reshape([n,1]) #I calculate every mi*mj pair, I'll get a matrix (n,n)

        #Now let's calculate the potential energy for every pair of particles:
        Epot = m_ij/(norm2_r_ij + softening**2.)**0.5

        #Extract upper and bottom triangular matrix with all the pairs
        Epot_up_list = Epot[np.triu_indices(n, k=1)]
        Epot_bottom_list = Epot[np.tril_indices(n, k=-1)]

        Epot_list = np.concatenate((Epot_up_list, Epot_bottom_list))

        #total potential energy
        Epot = -0.5*np.sum(Epot_list)

        return Epot
    
    def Etot_vett(self,softening: float = 0.) -> tuple[float,float,float]:
        """
        Estimate the total  energy of the particles: Etot=Ekintot + Epottot

        :param softening: Softening parameter
        :return: a tuple with

            - Total energy
            - Total kinetic energy
            - Total potential energy
        """

        Ekin = self.Ekin_vett()
        Epot = self.Epot_vett(softening=softening)
        Etot = Ekin + Epot

        return Etot, Ekin, Epot




