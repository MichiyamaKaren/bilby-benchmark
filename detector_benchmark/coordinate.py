# %%
from typing import Tuple
import numpy as np


class Coordinate3D:
    def __init__(self, x=None, y=None, z=None, r=None, theta=None, phi=None):
        if x is not None:
            assert y is not None
            assert z is not None
            self.cartesian = np.array([x, y, z])
        elif r is not None:
            assert theta is not None
            assert phi is not None
            self.spherical = np.array([r, theta, phi])
        self._rotation_matrix = np.eye(3)

    def cartesian_to_spherical(self, cartesian):
        x, y, z = cartesian
        r = np.linalg.norm(cartesian)
        theta = np.arccos(z/r)
        if x > 0:
            phi = np.arctan(y/x)
        elif x < 0:
            phi = np.arctan(y/x)+np.pi
        elif y > 0:
            phi = np.pi/2
        else:
            phi = np.pi*3/2
        phi %= 2*np.pi
        return np.array([r, theta, phi])

    def spherical_to_cartesian(self, spherical):
        r, theta, phi = spherical
        return np.array([r*np.sin(theta)*np.cos(phi),
                         r*np.sin(theta)*np.sin(phi),
                         r*np.cos(theta)])

    @property
    def cartesian(self):
        return np.matmul(self._rotation_matrix, self._cartesian)

    @cartesian.setter
    def cartesian(self, value):
        self._cartesian = value

    @property
    def spherical(self):
        return self.cartesian_to_spherical(self.cartesian)

    @spherical.setter
    def spherical(self, value):
        self._cartesian = self.spherical_to_cartesian(value)

    @property
    def r(self):
        return self.spherical[0]

    @r.setter
    def r(self, r_value):
        self.spherical = np.array([r_value, self.theta, self.phi])

    @property
    def theta(self):
        return self.spherical[1]

    @theta.setter
    def theta(self, theta_value):
        self.spherical = np.array([self.r, theta_value, self.phi])

    @property
    def phi(self):
        return self.spherical[2]

    @phi.setter
    def phi(self, phi_value):
        self.spherical = np.array([self.r, self.theta, phi_value])

    def rotate_x(self, angle):
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), np.sin(angle)],
            [0, -np.sin(angle), np.cos(angle)]])
        self._rotation_matrix = np.matmul(
            rotation_matrix, self._rotation_matrix)

    def rotate_y(self, angle):
        rotation_matrix = np.array([
            [np.cos(angle), 0, -np.sin(angle)],
            [0, 1, 0],
            [np.sin(angle), 0, np.cos(angle)]])
        self._rotation_matrix = np.matmul(
            rotation_matrix, self._rotation_matrix)

    def rotate_z(self, angle):
        rotation_matrix = np.array([
            [np.cos(angle), np.sin(angle), 0],
            [-np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]])
        self._rotation_matrix = np.matmul(
            rotation_matrix, self._rotation_matrix)

    def rotate_to(self, new_x, new_y):
        """
        rotate to a new coordinate frame whose x,y base vector is new_x,new_y
        Reference：《天体测量和天体力学基础》 附录(2.94)

        Args:
            new_x (1d array): new x-axis direction
            new_y (1d array): new y-axis direction
        """
        self.clear_rotation()

        new_z = np.cross(new_x, new_y)
        _, I, z_phi = self.cartesian_to_spherical(new_z)
        Omega = z_phi+np.pi/2
        u1 = np.array([np.cos(Omega), np.sin(Omega), 0])
        cos_omega = np.dot(new_x, u1) / np.linalg.norm(new_x)
        # handling possible float error on calculating cos_omega
        cos_omega = 1 if cos_omega > 1 else cos_omega
        cos_omega = -1 if cos_omega < -1 else cos_omega
        omega = np.arccos(cos_omega)

        self.rotate_z(Omega)
        self.rotate_x(I)
        self.rotate_z(omega)

    def inverse_rotation(self):
        self._rotation_matrix = np.linalg.inv(self._rotation_matrix)

    def clear_rotation(self):
        self._rotation_matrix = np.eye(3)


class CoordinateFrame:
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y

    def coordinate_transform(self, vector: np.ndarray) -> np.ndarray:
        """
        tranform a vector to this frame

        Args:
            vector (1d array): vector in original frame

        Returns:
            1d array: vector in this frame
        """
        coordinate = Coordinate3D(x=vector[0], y=vector[1], z=vector[2])
        coordinate.rotate_to(self.x, self.y)
        return coordinate.cartesian

    def coordinate_transform_spherical(self, theta: float, phi: float) -> Tuple[float, float]:
        """
        tranform a vector specified in spherical coordinate to this frame

        Args:
            theta (float): spherical coordinate theta
            phi (float): spherical coordinate phi

        Returns:
            Tuple[float,float]: transformed spherical coordinate theta,phi
        """
        coordinate = Coordinate3D(r=1, theta=theta, phi=phi)
        coordinate.rotate_to(self.x, self.y)
        return coordinate.theta, coordinate.phi

    def inverse_transform(self, vector: np.ndarray) -> np.ndarray:
        """
        tranform a vector in this frame to the original frame

        Args:
            vector (1d array): vector in this frame

        Returns:
            1d array: vector in original frame
        """
        coordinate = Coordinate3D(x=vector[0], y=vector[1], z=vector[2])
        coordinate.rotate_to(self.x, self.y)
        coordinate.inverse_rotation()
        return coordinate.cartesian

    def inverse_transform_spherical(self, theta: float, phi: float) -> Tuple[float, float]:
        """
        tranform a vector in this frame specified in spherical coordinate to the original frame

        Args:
            theta (float): spherical coordinate theta
            phi (float): spherical coordinate phi

        Returns:
            Tuple[float,float]: transformed spherical coordinate theta,phi in the original frame
        """
        coordinate = Coordinate3D(r=1, theta=theta, phi=phi)
        coordinate.rotate_to(self.x, self.y)
        coordinate.inverse_rotation()
        return coordinate.theta, coordinate.phi
