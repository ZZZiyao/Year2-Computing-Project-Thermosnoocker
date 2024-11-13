"""Simulation of collisions"""
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from thermosnooker.ballss import Ball, Container
from scipy.constants import Boltzmann


class Simulation:
    """This is a base class, code copied from script"""
    def next_collision(self):
        """
        Run the simulation for one collision.

        Args:
            None
        
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        """
        raise NotImplementedError('next_collision() needs to be implemented in derived classes')

    def setup_figure(self):
        """
        Setup the figure for the simulation.

        Args:
            None
        
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        """
        raise NotImplementedError('setup_figure() needs to be implemented in derived classes')

    def run(self, num_collisions, animate=False, pause_time=0.001):
        """
        Sequentially call the next_collision method to simulate each collision.

        Args:
            num_collisions (int): Number of collisions to simulate.
            animate (bool): If True, animate the simulation.
            pause_time (float): Pause time between frames in the animation.
        """
        if animate:
            self.setup_figure()
        for _ in range(num_collisions):
            self.next_collision()
            if animate:
                plt.pause(pause_time)
        if animate:
            plt.show()

class SingleBallSimulation(Simulation):
    """Simulate with one ball and one container """
    def __init__(self, container, ball):
        """
        Initialize the SingleBallSimulation.

        Args:
            container (Container): The container object.
            ball (Ball): The ball object.
        """
        self._container = container
        self._ball = ball

    def container(self):
        """
        Return the Container object.

        Args:
            None

        Returns:
            Container: The container object.
        """
        return self._container

    def ball(self):
        """
        Return the Ball object.

        Args:
            None

        Returns:
            Ball: The ball object.
        """
        return self._ball

    def setup_figure(self):
        """
        Copied from script
        
        Args:
            None
        """
        rad = self.container().radius()
        fig = plt.figure()
        ax = plt.axes(xlim=(-rad, rad), ylim=(-rad, rad))
        ax.add_artist(self.container().patch())
        ax.add_patch(self.ball().patch())

    def next_collision(self):
        """
        Calculate when the next collision will occur and between which objects.
        Move the ball to this point in time.
        Perform the elastic collision.

        Args:
            None
        """
        # time of next collision
        ttc = self._container.time_to_collision(self._ball)
        if ttc is None:  # end running if no collision
            return

        # new positions of ball after ttc
        self._ball.move(ttc)

        # perform collision
        self._ball.collide(self._container)


def rtrings(rmax, nrings, multi):
    """
    Generate position of points in a circle ring.

    Args:
        rmax (float): Maximum radius of the ring of balls.
        nrings (int): Number of rings of balls.
        multi (int): Multiplier for each successive ring.

    Yields:
        tuple: The position of points in a circle ring.
    """
    yield [0, 0]
    rstep = rmax / nrings

    for rings in range(1, nrings + 1):
        angstep = 2 * np.pi / (multi * rings)
        radius = rings * rstep

        for points in range(multi * rings):  # calculate theta
            angle = points * angstep
            yield (radius * np.cos(angle), radius * np.sin(angle))


class MultiBallSimulation(Simulation):
    """Handle multiple balls."""
    def __init__(self, c_radius=10., b_radius=1., b_speed=10., b_mass=1., rmax=8., nrings=3, multi=6,cor=1.0):
        """
        Initialize the MultiBallSimulation with the container and balls.

        Args:
            c_radius (float): Container radius.
            b_radius (float): Ball radius.
            b_speed (float): Ball speed.
            b_mass (float): Ball mass.
            rmax (float): Maximum radius of the ring of balls.
            nrings (int): Number of rings of balls.
            multi (int): Multiplier for each successive ring.
        """
        self._time = 0  # for later uses
        self._container = Container(radius=c_radius)
        self._balls = []
        self._b_mass=b_mass
        self._rmax=rmax
        self._nrings=nrings
        self._multi=multi
        self._cor=cor
        self._collision_times = []
        for pos in rtrings(rmax, nrings, multi):  # positions of balls in the rings
            random_direction = random.uniform(0.0, 2 * np.pi)  # [0,2pi) polar coordinate
            xy_vel = [b_speed * np.cos(random_direction), b_speed * np.sin(random_direction)]  # in xy coordinate
            balls = Ball(pos=pos, vel=xy_vel, radius=b_radius, mass=b_mass)
            self._balls.append(balls)
        self._initial_energy = self.total_energy()

    def container(self):
        """
        Return the container object.

        Args:
            None

        Returns:
            Container: The container object.
        """
        return self._container

    def balls(self):
        """
        Return a list of all balls in the simulation.

        Args:
            None

        Returns:
            list: A list of Ball objects.
        """
        return self._balls
    
    def ball_mass(self):
        """
        Return the mass of the balls in the simulation.

        Args:
            None

        Returns:
            float: The mass of the balls.
        """
        return self._b_mass
    
    def num_of_balls(self):
        """
        Return the number of balls in the simulation.
        The number of balls is calculated based on the maximum radius of the rings,
        the number of rings, and the multiplier for each successive ring.

        Args:
            None

        Returns:
            int: The number of balls in the simulation.
        """
        num_balls = len(list(rtrings(self._rmax,self._nrings,self._multi)))
        return num_balls

    def setup_figure(self):
        """
        Setup the figure for the simulation.

        Args:
            None
        """
        rad = self.container().radius()
        fig = plt.figure()
        ax = plt.axes(xlim=(-rad, rad), ylim=(-rad, rad))
        ax.add_artist(self.container().patch())
        for ball in self._balls:
            ax.add_patch(ball.patch())

    def next_collision(self):
        """
        Calculate when the next collision will occur and between which objects.
        Move all balls to this point in time.
        Perform the elastic collision.

        Args:
            None
        """
        # first consider ball ball collisions
        next_collision = float('inf')  # set to infinity at first, whenever there is a smaller ttc it will update
        collision_obj1 = None
        collision_obj2 = None
        for i in range(len(self._balls)):  # number of balls
            for j in range(i + 1, len(self._balls)):  # each ball other than ith ball collide with ith ball
                ttc = self._balls[i].time_to_collision(self._balls[j])
                if ttc is not None and 0 < ttc < next_collision:
                    next_collision = ttc  # if there is a smaller t, update it
                    collision_obj1 = self._balls[i]  # update the corresponding collision pair
                    collision_obj2 = self._balls[j]

        # next we consider ball container collision
        for i in range(len(self._balls)):  # number of balls
            ttc = self._container.time_to_collision(self._balls[i])
            if ttc is not None and 0 < ttc < next_collision:
                next_collision = ttc
                collision_obj1 = self._balls[i]  # update the corresponding collision pair
                collision_obj2 = self._container

        if next_collision == float('inf'):
            return

        # move the balls to the position when next collision occur
        for ball in self._balls:
            ball.move(next_collision)

        # collide the colliding objects
        collision_obj2.collide(collision_obj1,restitution=self._cor)

        # add time of running
        self._time += next_collision
        self._collision_times.append(self._time)

    def set_restitution(self, cor):
        """Set the coefficient of restitution for collisions"""
        self._cor = cor


    def kinetic_energy_old(self):
        """
        Calculate and return the total kinetic energy of the system.

        Args:
            None

        Returns:
            float: Total kinetic energy of the system.
        """
        ke_balls = sum(0.5 * ball.mass() * np.dot(ball.vel(), ball.vel()) for ball in self._balls)
        ke_container = 0.5 * self._container.mass() * np.dot(self._container.vel(), self._container.vel())
        return ke_balls + ke_container

    def momentum(self):
        """
        Calculate and return the vectoral sum of all momenta in the system.

        Args:
            None

        Returns:
            numpy.ndarray: Total momentum of the system.
        """
        momentum_balls = sum(ball.mass() * ball.vel() for ball in self._balls)
        momentum_container = self._container.mass() * self._container.vel()
        return momentum_balls + momentum_container

    def time(self):
        """
        Returns the current time the simulation has been running for.

        Args:
            None

        Returns:
            float: The current time.
        """
        return self._time

    def pressure(self):
        """
        Calculate and return the pressure of the system.

        Args:
            None

        Returns:
            float: The pressure of the system.
        """
        return self._container.dp_tot() / (self._time * self._container.surface_area())
    
    def t_equipartition(self):
        """
        Return the temperature based on the theory of equipartition of energy.

        Args:
            None

        Returns:
            float: The temperature of the system.
        """
        n=2
        ke_theory=self.kinetic_energy()/self.num_of_balls()
        T_theory=2*ke_theory/(n*Boltzmann)
        return T_theory
    
    def t_ideal(self):
        """
        Return the temperature based on the theory of ideal gas law.

        Args:
            None

        Returns:
            float: The temperature of the system.
        """
        n=self.num_of_balls()
        v=self.container().volume()
        p=self.pressure()
        t_ideal=p*v/(Boltzmann*n)
        return t_ideal

    def speeds(self):
        """
        Return a list of the speeds of all balls in the simulation.

        Args:
            None

        Returns:
            list: A list of speeds of all balls.
        """
        speeds=[]
        for ball in self.balls():
            speed=norm(ball.vel())
            speeds.append(speed)  
        return speeds
    
    def total_energy(self):
        ke = [0.5 * ball.mass() * norm(ball.vel()) ** 2 for ball in self.balls()]
        ke_container = 0.5 * self._container.mass() * norm(self._container.vel()) ** 2
        return sum(ke) + ke_container

    def kinetic_energy(self):
        return self.total_energy()

    

    def calculate_entropy_the(self):
        """Calculate the entropy of the system."""
        N = len(self._balls)  # Number of particles
        V = self._container.volume()  # Volume of the container (area in 2D)
        U = self.kinetic_energy_old()  # Total kinetic energy (internal energy in 2D)
        m = self._balls[0].mass()  # Mass of each particle (assuming all have the same mass)
        k_B = 1.380649e-23  # Boltzmann constant in J/K
        h = 6.62607015e-34  # Planck constant in J*s
        
        # Calculate entropy
        entropy = N * k_B * (np.log(V / N * (2 * np.pi * m * U / (N * h**2))) + 2.5)
        print(U)
        return entropy
    
    def temperature(self):
        k_b = 1.38e-23  # Boltzmann constant
        total_ke = self.kinetic_energy()
        n_particles = len(self._balls) + 1  # Balls + container
        return (2 / 3) * (total_ke / (n_particles * k_b))

    def calculate_entropy(self):
        energies = [0.5 * ball.mass() * np.dot(ball.vel(), ball.vel()) for ball in self.balls()]
        energies.append(0.5 * self._container.mass() * np.dot(self._container.vel(), self._container.vel()))
        k_b = 1.38e-23  # Boltzmann constant
        probabilities = np.array(energies) / np.sum(energies)
        entropy = -k_b * np.sum(probabilities * np.log(probabilities))
        return entropy

    def environment_entropy_change(self, T_env):
        current_energy = self.total_energy()
        energy_change = current_energy - self._initial_energy
        return energy_change / T_env

    def total_entropy(self, T_env):
        system_entropy = self.calculate_entropy()
        env_entropy_change = self.environment_entropy_change(T_env)
        return system_entropy + env_entropy_change
    
    def calculate_entropy_the_2d(self):
        """Calculate the entropy of the system in 2D."""
        N = len(self._balls)  # Number of particles
        A = self._container.volume()  # Area of the container (2D volume)
        U = self.kinetic_energy_old()  # Total kinetic energy
        m = self._balls[0].mass()  # Mass of each particle (assuming all have the same mass)
        k_B = 1.380649e-23  # Boltzmann constant in J/K
        h = 6.62607015e-34  # Planck constant in J*s
        
        # Calculate temperature
        T = (2 * U) / (N * k_B)
        
        # Calculate entropy using the 2D Sackur-Tetrode equation
        entropy = N * k_B * (np.log(A / N * (2 * np.pi * m * k_B * T / h**2)) + 1)
        return entropy
    
    def collision_times(self):
        return self._collision_times
    
    def average_time_between_collisions(self):
        """Calculate the average time between collisions."""
        if len(self._collision_times) < 2:
            return float('inf')
        intervals = np.diff(self._collision_times)
        average_interval = np.mean(intervals)
        return average_interval
