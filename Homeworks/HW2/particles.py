import numpy as np
import numpy.typing as npt

class Particles:
    def __init__(self, position: npt.NDArray[np.float64], velocity: npt.NDArray[np.float64], mass: npt.NDArray[np.float64]):
        """
        Class initialiser.
        It assigns the values to the class member pos, vel, mass and ID.
        ID is just a sequential integer number associated to each particle.

        :param position: A Nx3 numpy array containing the positions of the N particles
        :param velocity: A Nx3 numpy array containing the velocity of the N particles
        :param mass: A Nx1 numpy array containing the mass of the N particles
        """

        self.pos = np.array(np.atleast_2d(position), dtype=float)
        if self.pos.shape[1] < 2: print(f"Input position should contain at leat a Nx2 array, current shape is {self.pos.shape}")

        self.vel = np.array(np.atleast_2d(velocity), dtype=float)
        if self.vel.shape[1] < 2: print(f"Input velocity should contain at leat a Nx2 array, current shape is {self.pos.shape}")
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
        if acceleration.shape[1] < 2: print(f"Input acceleration should contain a Nx3 array, current shape is {acc.shape}")

        self.acc=acc

    def com_pos(self) -> npt.NDArray[np.float64]:
        """
        Estimate the position of the centre of mass

        :return: a numpy  array with three elements corresponding to the centre of mass position
        """

        return np.sum(self.mass*self.pos.T,axis=1)/np.sum(self.mass)


