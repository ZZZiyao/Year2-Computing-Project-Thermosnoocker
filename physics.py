"""Maxwell Boltzmann PDF"""
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from thermosnooker.simulationss import MultiBallSimulation

def maxwell(speed, kbt, mass=1.):
    """
    Return the Maxwell-Boltzmann PDF value at a given speed.

    Args:
        speed (float): Speed value.
        kbt (float): Product of Boltzmann constant and temperature.
        mass (float): Mass of the particles, default is 1.0.

    Returns:
        float: Maxwell-Boltzmann PDF value.
    """
    normfac = mass / kbt #normalization factor
    exponent = np.exp(-mass * speed**2 / (2 * kbt))
    return normfac*speed*exponent

# Constants
k_B = 1.380649e-23  # Boltzmann constant in J/K
h = 6.62607015e-34  # Planck constant in J*s

mbs=MultiBallSimulation(b_radius=0.1,cor=0.9)

# Parameters for Argon gas
N = mbs.num_of_balls()   # Number of particles
V = mbs.container().volume()  # Volume in m^3
T = mbs.t_equipartition()  # Temperature in K
m = mbs.ball_mass()  # Mass of one Argon atom in kg
d = 0.2  # Diameter of an Argon atom in meters

# Calculate the effective collision cross-section
sigma = 2*d

# Calculate the number density
n = N / V

# Calculate the mean free path
lambda_mean_free_path = 1 / (np.sqrt(2) * n * sigma)

# Calculate the average speed in 2D
average_speed = np.sqrt((2 * k_B * T) / m)

# Calculate the average collision time
average_collision_time = lambda_mean_free_path / 5.4

# Print results
print(f"Effective collision cross-section (σ): {sigma:.3e} m^2")
print(f"Number density (n): {n:.3e} particles/m^2")
print(f"Mean free path (λ): {lambda_mean_free_path:.3e} m")
print(f"Average speed (⟨v⟩): {average_speed:.3e} m/s")
print(f"Average collision time (τ): {average_collision_time:.3e} s")

