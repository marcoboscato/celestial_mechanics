import numpy as np
import numpy.typing as npt
__all__ = ['Moons']

class Moons:

    def __init__(self, position: npt.NDArray[np.float64], velocity: npt.NDArray[np.float64], mass: npt.NDArray[np.float64]):
        """
        Class initialiser:
        param position: A Nx3 numpy array containing the positions of the N particles
        param velocity: A Nx3 numpy array containing the velocity of the N particles
        param mass: A Nx1 numpy array containing the mass of the N particles
        """
        self.pos = np.array(np.atleast_2d(position), dtype=float)
        if self.pos.shape[1] != 3: print(f"Input position should contain a Nx3 array, current shape is {self.pos.shape}")

        self.vel = np.array(np.atleast_2d(velocity), dtype=float)
        if self.vel.shape[1] != 3: print(f"Input velocity should contain a Nx3 array, current shape is {self.pos.shape}")
        if len(self.vel) != len(self.pos): print(f"Position and velocity in input have not the same number of elemnts")

        self.mass = np.array(np.atleast_1d(mass), dtype=float)
        if len(self.mass) != len(self.pos): print(f"Position and mass in input have not the same number of elemnts")

        self.ID=np.arange(len(self.mass), dtype=int)

        self.acc=None


    def radius(self) -> npt.NDArray[np.float64]:
        """
        Estimate the particles distance from the origin of the frame of reference.

        :return:  a Nx1 array containing the particles' distance from the origin of the frame of reference.
        """

        return np.sqrt(np.sum(self.pos*self.pos, axis=1))[:,np.newaxis]

    def separation(self) -> float:
        """
        Estimate the distance between the 2 moons.

        :return: the distance between the 2 moons
        """

        return np.sqrt(np.sum((self.pos[1]-self.pos[2])**2.))

    
    def SysEkin(self) -> float:
        """
        Estimate the total potential energy of the system:
        Ekin=0.5 sum_i mi vi*vi

        :return: total kinetic energy
        """
        Ekin = 0.0
        for i in range(len(self.mass)):
            vel_sq = sum(v**2. for v in self.vel[i])
            Ekin_particle = self.mass[i] * vel_sq
            Ekin += Ekin_particle

        return 0.5 * Ekin
    

    def SysEpot(self, softening: float = 0.) -> float:
        """
        Estimate the total potential energy of the system:
        Epot=-0.5 sumi sumj mi*mj / sqrt(rij^2 + eps^2)
        where eps is the softening parameter

        :param softening: Softening parameter
        :return: The total potential energy of the system
        """
        Epot = 0.0
        for i in range(len(self.mass)):
            for j in range(len(self.mass)):
                if i != j:
                    rij_sq = sum((a-b)**2. for a, b in zip(self.pos[i],self.pos[j]))
                    rij = np.sqrt(rij_sq + softening**2.)
                    Epot_particle = (self.mass[i] * self.mass[j]) / (rij)
                    Epot += Epot_particle

        return - 0.5 * Epot
    
    
    def Etot(self,softening: float = 0.) -> tuple[float,float,float]:
        """
        Estimate the total  energy of the particles: Etot=Ekintot + Epottot

        :param softening: Softening parameter
        :return: a tuple with

            - Total energy
            - Total kinetic energy
            - Total potential energy
        """
        Ekin = self.SysEkin()
        Epot = self.SysEpot(softening=softening)
        Etot = Ekin + Epot

        return Etot, Ekin, Epot
    

    def copy(self):# -> Moons:
        """
        Return a copy of this Particle class

        :return: a copy of the Particle class
        """

        moon=Moons(np.copy(self.pos),np.copy(self.vel),np.copy(self.mass))
        if self.acc is not None: moon.acc=np.copy(self.acc)

        return Moons(np.copy(self.pos),np.copy(self.vel),np.copy(self.mass))

    def __len__(self) -> int:
        """
        Special method to be called when  this class is used as argument
        of the Python built-in function len()
        :return: Return the number of particles
        """

        return len(self.mass)

    def __str__(self) -> str:
        """
        Special method to be called when  this class is used as argument
        of the Python built-in function print()
        :return: short info message
        """

        return f"Instance of the class Particles\nNumber of particles: {self.__len__()}"
    

    