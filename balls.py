"""Create ball and container class"""
import numpy as np
from matplotlib.patches import Circle
from numpy.linalg import norm

TOLERANCE = 1e-9  # Define a small tolerance for floating-point comparisons
MIN_TIME = 1e-6  # Define a minimum time threshold to avoid unrealistic small collision times

class Ball:
    """ball class"""
    def __init__(self, pos=[0.0,0.0], vel=[1.0,0.0], radius=1.0, mass=1.0):
        """
        Initialize a Ball object.

        Args:
            pos (list): Ball's position, default [0.0, 0.0].
            vel (list): Ball's velocity, default [1.0, 0.0].
            radius (float): Ball's radius, default 1.0.
            mass (float): Ball's mass, default 1.0.

        Raises:
            ValueError: if length of position and/or velocity is not 2
        """
        self._pos = np.array(pos, dtype=np.float64)
        if len(self._pos) != 2:
            raise ValueError("Length of pos must be exactly 2") #ensure length of pos
        self._vel = np.array(vel, dtype=np.float64)
        if len(self._vel) != 2:
            raise ValueError("Length of vel must be exactly 2") #ensure length of vel
        self._radius = float(radius)
        self._mass = float(mass)
        self._patch = Circle((self._pos[0], self._pos[1]), self._radius)
        self._dp_tot=0.0 #for later uses

    def pos(self):
        """
        Return the current position of the centre of the ball.

        Args:
            None

        Returns:
            numpy.ndarray: The position of the ball.
        """
        return self._pos

    def radius(self):
        """
        Return the current radius of the ball.

        Args:
            None

        Returns:
            float: The radius of the ball.
        """
        return self._radius

    def mass(self):
        """
        Return the current mass of the ball.

        Args:
            None

        Returns:
            float: The mass of the ball.
        """
        return self._mass

    def vel(self):
        """
        Return the current velocity of the ball.

        Args:
            None

        Returns:
            numpy.ndarray: The velocity of the ball.
        """
        return self._vel

    def set_vel(self, vel):
        """
        Set the velocity of the ball to a new value.

        Args:
            vel (list or numpy.ndarray): The new velocity of the ball.

        Raises:
            ValueError:If length of velocity is not 2
        """
        self._vel = np.array(vel)
        if len(self._vel) != 2:
            raise ValueError("Length of vel must be exactly 2") #ensure length of vel
        
    def patch(self):
        """
        Return the patch of the ball.

        Args:
            None

        Returns:
            matplotlib.patches.Circle: The patch of the ball.
        """
        return self._patch

    def time_to_collision(self, other):
        """
        Calculate the time to collision with another object.

        Args:
            other: Another ball or the container.

        Returns:
            float or None: The time to collision, or None if no collision occurs.

        Raises:
            TypeError: If the 'other' parameter is not a Ball or Container object.
        """
        if isinstance(other, Ball): # check if the other parameteris an instance of the Ball class.
            r = self._pos - other.pos()
            v = self._vel - other.vel()
            if other.radius() == self.radius():
                relr = self._radius + other.radius()  # ball-ball collision
            elif other.radius() > self.radius():
                relr = other.radius() - self._radius  # ball-container collision (self is smaller)
            else:
                relr = self._radius - other.radius() #other is smaller
            #solve the quadratic equation
            a = np.dot(v, v)
            b = 2 * np.dot(r, v)
            c = np.dot(r, r) - relr**2
            #print("b=", b, r, v)
            # if a == 0: #parallel movement
            #     return None
            if np.isclose(a, 0, atol=TOLERANCE):  # Parallel movement
                return None
            discriminant = b**2 - 4 * a * c
            #print(discriminant, type(discriminant), R, a, b, c)
            if discriminant < 0: #complex solution
                return None
            t1 = (-b + np.sqrt(discriminant)) / (2 * a)
            t2 = (-b - np.sqrt(discriminant)) / (2 * a)
            if t1 < 0 and t2 < 0:
                return None
            if t1 > 0 and t2 > 0:
                t_min = min(t1, t2)
                if t_min < MIN_TIME:  # avoid float pointing errors
                    return None
                return t_min  # Return earlier time
            if t1 > 0:
                if t1 < MIN_TIME:  # Check for extremely small times (float pointing error)
                    return None
                return t1
            if t2 > 0:
                if t2 < MIN_TIME:  
                    return None
                return t2
            return None  # When t1 == t2, no collision because balls only "kiss"
        else:
            raise TypeError("The 'other' parameter must be a Ball object.")
        
    def move(self, dt):
        """
        Move the ball to a new position.

        Args:
            dt (float): Time interval over which to move the ball.
        """
        self._pos += self._vel * dt  #Formula:r' = r + v * dt
        self._patch.center = self._pos

    def collide(self,other):
        """
        Calculate the new velocities after collision with another Ball object.

        Args:
            other (Ball): Another ball object.

        Raises:
            TypeError: If the 'other' parameter is not a Ball object.
        """
        if isinstance(other,Ball):
            m1, m2 = self.mass(), other.mass()
            u1, u2 = self.vel(), other.vel()
            r1, r2 = self.pos(), other.pos()
            # Calculate normal and tangential velocity components of balls
            normal = r2 - r1 
            tangent = np.array([-normal[1], normal[0]]) #tangent and normal dot to zero
            #normalization 
            normal /= norm(normal)
            tangent /= norm(tangent)
            # Compute velocities in the normal and tangential coordinates
            u1_n=np.dot(u1, normal)
            u1_t=np.dot(u1, tangent)
            u2_n=np.dot(u2, normal)
            u2_t=np.dot(u2, tangent)
            # Calculate new normal velocities after collision using the formulas for elastic collision
            v1_n = (u1_n * (m1 - m2) + 2 * m2 * u2_n) / (m1 + m2)
            v2_n = (u2_n * (m2 - m1) + 2 * m1 * u1_n) / (m1 + m2)
            # Tangential velocities remain unchanged
            v1_t=u1_t
            v2_t=u2_t
            # Combine the normal and tangential components
            new_vel1 = v1_n*normal + v1_t*tangent
            new_vel2 = v2_n*normal + v2_t*tangent
            # Update the velocities of the balls
            self.set_vel(new_vel1)
            other.set_vel(new_vel2)
            # For later uses, we calculate the change in momentum of ball 2 as well 
            impulse = (v1_n-u1_n)*m1*normal #in the direction of normal, change in momentum is 2mv
            self._dp_tot+=norm(impulse)
        else:
            raise TypeError("The 'other' parameter must be a Ball object.")
        
    def dp_tot(self):
        """
        Return the running total of the change in momentum imparted upon the container.

        Args:
            None

        Returns:
            float: The total change in momentum.
        """
        return self._dp_tot
    
class Container(Ball):
    """container class"""
    def __init__(self,radius=10.0,mass=10000000.0):
        """
        Initialize a container object.

        Args:
            radius (float): Container's radius, default 10.0.
            mass (float): Container's mass, default 10000000.0.
        """
        super().__init__(pos=[0.0, 0.0], vel=[0.0, 0.0], radius=radius, mass=mass)
        #self._dp_tot=0.0
        self._patch.set_fill(False) #its color should not cover the balls

    def volume(self):
        """
        Return the volume the container (area in 2D).

        Args:
            None

        Returns:
            float: The volume of the container.
        """
        return (self._radius**2)*np.pi

    def surface_area(self):
        """
        Return the surface area of the container (circumference in 2D).

        Args:
            None

        Returns:
            float: The surface area of the container.
        """
        return 2*np.pi*self._radius
    