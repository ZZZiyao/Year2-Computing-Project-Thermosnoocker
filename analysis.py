"""Analysis Module."""
import matplotlib.pyplot as plt
import numpy as np
from thermosnooker.balls import Container, Ball
from thermosnooker.simulations import SingleBallSimulation
from thermosnooker.simulations import MultiBallSimulation
import itertools
from numpy.linalg import norm
from thermosnooker.physics import maxwell
from scipy.constants import Boltzmann

def task9():
    """
    Task 9.

    In this function, you should test your animation. To do this, create a container
    and ball as directed in the project brief. Create a SingleBallSimulation object from these
    and try running your animation. Ensure that this function returns the balls final position and
    velocity.

    Args:
        

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: The balls final position and velocity
    """
    c = Container(radius=10.)
    b = Ball(pos=[-5, 0], vel=[1, 0.], radius=1., mass=1.)
    sbs = SingleBallSimulation(container=c, ball=b)
    #run simulation using the run() method for 20 collisions with the animation turned on and the pause_time set to 0.5
    #def run(self, num_collisions, animate=False, pause_time=0.001)
    sbs.run(num_collisions=20, animate=True, pause_time=0.5)
    #find final position and velocity of ball
    final_state=[b.pos(),b.vel()]
    final_state=np.array(final_state,dtype=np.float64)

    return final_state

def task10():
    """
    Task 10.
    In this function we shall test your MultiBallSimulation. Create an instance of this class using
    the default values described in the project brief and run the animation for 500 collisions.
    Watch the resulting animation carefully and make sure you aren't seeing errors like balls sticking
    together or escaping the container.

    Args:
        None
    """
    mbs = MultiBallSimulation()
    #run simulation using the run() method for 20 collisions with the animation turned on and the pause_time set to 0.5
    #def run(self, num_collisions, animate=False, pause_time=0.001)
    mbs.run(num_collisions=500, animate=True)

def task11():
    """
    Task 11.
    In this function we shall be quantitatively checking that the balls aren't escaping or sticking.
    To do this, create the two histograms as directed in the project script. Ensure that these two
    histogram figures are returned.

    Args:
        None

    Returns:
        tuple[Figure, Firgure]: The histograms (distance from centre, inter-ball spacing).
    """
    mbs=MultiBallSimulation()
    mbs.run(num_collisions=500,animate=False)
    balls=mbs.balls() #these are the balls
    container=mbs.container()
    #first consider ball ball
    balls_distances=[]
    for ball1, ball2 in itertools.combinations(balls, 2):
        distance=norm(ball1.pos()-ball2.pos())
        balls_distances.append(distance)
    #and then we consider ball container 
    bc_distances=[]
    for ball in balls:
        distance=norm(ball.pos()-container.pos())
        bc_distances.append(distance)
    #create histogram for ball container distances
    plt.figure()
    plt.hist(bc_distances, bins=10)
    plt.title('Distance from Center')
    plt.xlabel('Distance (m)')
    plt.ylabel('Frequency')
    plt.grid()
    fig1 = plt.gcf()  
    # Create histogram for ball ball distances
    plt.figure()
    plt.hist(balls_distances, bins=50)
    plt.title('Inter-Ball Distances')
    plt.xlabel('Distance (m)')
    plt.ylabel('Frequency')
    plt.grid(True)
    fig2 = plt.gcf()  
    plt.show()
    return fig1,fig2

task11()
def task12():
    """
    Task 12.
    In this function we shall check that the fundamental quantities of energy and momentum are conserved.
    Additionally we shall investigate the pressure evolution of the system. Ensure that the 4 figures
    outlined in the project script are returned.

    Args:
        None

    Returns:
        tuple[Figure, Figure, Figure, Figure]: matplotlib Figures of the KE, momentum_x, momentum_y ratios
        as well as pressure evolution.
    """
    mbs = MultiBallSimulation(c_radius=10., b_radius=1., b_speed=10., b_mass=1,rmax=8., nrings=3, multi=6)
    num_collisions=1000
    #before running next_collision, let's find initial momen and KE
    momen0=mbs.momentum()
    ke0=mbs.kinetic_energy()
    momen_x0=momen0[0]
    momen_y0=momen0[1]
    times=[]#find array of collision times
    energies=[]#find array of KE at time t
    ke_ratio=[]#find array of KE(t)/KE(0)
    momen_xratio=[]#array of momentum x component ratio
    momen_yratio=[]#array of momentum y component ratio
    pressures=[]#array of pressure
    for _ in range(num_collisions):
        mbs.next_collision()
        times.append(mbs.time())
        energies.append(mbs.kinetic_energy())
        ke_ratio.append(mbs.kinetic_energy()/ke0)
        momentum=mbs.momentum()
        momen_xratio.append(momentum[0]/momen_x0)
        momen_yratio.append(momentum[1]/momen_y0)
        pressures.append(mbs.pressure())
    #create histogram for KE(t)/KE(0) against t
    plt.figure()
    plt.plot(times,ke_ratio)
    plt.title('KE(t)/KE(0) vs t')
    plt.xlabel('time')
    plt.ylabel('KE(t)/KE(0)')
    plt.grid()
    fig3 = plt.gcf()  
    # Create histogram for momentum x ratio
    plt.figure()
    plt.plot(times,momen_xratio)
    plt.title('Σmomentum_x(t) / Σmomentum_x(0) vs t')
    plt.xlabel('time')
    plt.ylabel('Σmomentum_x(t) / Σmomentum_x(0)')
    plt.grid()
    fig4 = plt.gcf()  
    # Create histogram for momentum y ratio
    plt.figure()
    plt.plot(times,momen_yratio)
    plt.title('Σmomentum_y(t) / Σmomentum_y(0) vs t')
    plt.xlabel('time')
    plt.ylabel('Σmomentum_y(t) / Σmomentum_y(0)')
    plt.grid()
    fig5 = plt.gcf()
    # Create histogram for pressure
    plt.figure()
    plt.plot(times,pressures)
    plt.title('pressure vs t')
    plt.xlabel('time')
    plt.ylabel('pressure')
    plt.grid()
    fig6 = plt.gcf()
    plt.show()
    return fig3,fig4,fig5,fig6

def task13():
    """
    Task 13.
    In this function we investigate how well our simulation reproduces the distributions of the IGL.
    Create the 3 figures directed by the project script, namely:
    1) PT plot
    2) PV plot
    3) PN plot
    Ensure that this function returns the three matplotlib figures.

    Args:
        None

    Returns:
        tuple[Figure, Figure, Figure]: The 3 requested figures: (PT, PV, PN)
    """
    #PT plot
    speeds=np.linspace(0.1,300,10)
    temperatures=[]
    pressures_tem=[]
    for speed in speeds:
        mbs = MultiBallSimulation(b_speed=speed,b_radius=0.1)
        mbs.run(num_collisions=500,animate=False)#pressure should stablize
        mbs.t_equipartition() 
        temperatures.append(mbs.t_equipartition()) #tem of this speed
        pressures_tem.append(mbs.pressure())#stabled pressure of this speed
    # ideal gas prediction
    kb=1.38e-23
    mbs1=MultiBallSimulation(b_radius=0.1)
    volume=mbs1.container().volume()
    ig_pressures = [(mbs1.num_of_balls() * kb * T) / volume for T in temperatures] #PV=NkT
    #plotting
    fig7 = plt.figure()
    plt.plot(temperatures, pressures_tem, 'x-', color='blue',label='simulation points')
    plt.plot(temperatures, ig_pressures, 'x--', color='red',label='idea gas prediction')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Pressure (Pa)')
    plt.title('Pressure vs Temperature')
    plt.grid()
    plt.legend()
    #return fig7
    #PV plot
    volumes = []
    pressures_vo = []
    radius = np.linspace(10, 20, 20)
    for r in radius:
        mbs = MultiBallSimulation(c_radius=r, b_radius=0.1)
        mbs.run(num_collisions=500, animate=False) #after p stablize
        volumes.append(mbs.container().volume())
        pressures_vo.append(mbs.pressure()) 
        print (r)
    fig8 = plt.figure()
    plt.plot(volumes, pressures_vo, 'x-', label='simulation points')
    plt.xlabel('Volume (m^3)')
    plt.ylabel('Pressure (Pa)')
    plt.title('Pressure vs Volume')
    plt.grid()
    plt.legend()
    # PN plot
    num_balls = []
    pressures_num = []
    nrings=range(1,8)
    for n in nrings:
        mbs = MultiBallSimulation(nrings=n,b_radius=0.1)
        mbs.run(num_collisions=500, animate=False) #after p stablize
        num_balls.append(mbs.num_of_balls())
        pressures_num.append(mbs.pressure())
        print(n)
    fig9 = plt.figure()
    plt.plot(num_balls, pressures_num, 'x-', label='simulation points')
    plt.xlabel('Number of Balls')
    plt.ylabel('Pressure (Pa)')
    plt.title('Pressure vs Number of Balls')
    plt.grid()
    plt.legend()
    return fig7, fig8, fig9

def task14():
    """
    Task 14.
    In this function we shall be looking at the divergence of our simulation from the IGL. We shall
    quantify the ball radii dependence of this divergence by plotting the temperature ratio defined in
    the project brief.

    Args:
        None

    Returns:
        Figure: The temperature ratio figure.
    """
    radius=np.linspace(0.01,1,20)
    T_ratio=[] #partition/ideal
    for r in radius:
        mbs=MultiBallSimulation(b_radius=r)
        mbs.run(num_collisions=1000,animate=False)#wait it to be stable
        ratio=mbs.t_equipartition()/mbs.t_ideal()
        T_ratio.append(ratio)
        print(r)
    fig10 = plt.figure()
    plt.plot(radius, T_ratio, 'x-')
    plt.xlabel('radius of balls (m)')
    plt.ylabel('T-equapartition/T-ideal')
    plt.title('Divergence of simulation from IGL')
    plt.grid()
    plt.show()
    return fig10

def task15():
    """
    Task 15.
    In this function we shall plot a histogram to investigate how the speeds of the balls evolve from the initial
    value. We shall then compare this to the Maxwell-Boltzmann distribution. Ensure that this function returns
    the created histogram.

    Args:
        None

    Returns:
        Figure: The speed histogram.
    """
    speeds_to_simulate = [10.0, 20.0, 30.0]
    colors = ['g', 'b', 'r']
    labels = ['Simulation 10 m/s', 'Simulation 20 m/s', 'Simulation 30 m/s']
    labels2=['Boltzmann 10m/s','Boltzmann 20m/s','Boltzmann 30m/s']
    
    plt.figure()

    for initial_speed, color, label1,label2 in zip(speeds_to_simulate, colors, labels,labels2):
        mbs = MultiBallSimulation(b_speed=initial_speed)
        mbs.run(1000)  # Run the simulation until it stabilizes

        speeds = mbs.speeds()
        t_ideal = mbs.t_ideal()
        mass = mbs.ball_mass()
        kbt = Boltzmann * t_ideal

        # Plot the histogram of speeds
        plt.hist(speeds, bins=30, density=True, alpha=0.6, color=color, label=label1)

        # Calculate the Maxwell-Boltzmann distribution
        speed_range = np.linspace(0, max(speeds), 100)
        mb_distribution = [maxwell(s, kbt, mass) for s in speed_range]
        print(initial_speed)

        # Plot the Maxwell-Boltzmann distribution
        plt.plot(speed_range, mb_distribution, color=color, linestyle='dashed',label=label2)

    plt.xlabel('Speed')
    plt.ylabel('Probability Density')
    plt.title('Speed Distribution')
    plt.legend()
    plt.grid()

    fig11 = plt.gcf()
    plt.show()

    return fig11


# def run_simulation(num_collisions):
#     mbs=MultiBallSimulation()
#     entropies = []
#     times = []
    
#     for _ in range(num_collisions):
#         mbs.next_collision()
#         current_time = mbs.time()
#         current_entropy = mbs.calculate_entropy()
        
#         times.append(current_time)
#         entropies.append(current_entropy)

#         print(_)
    
#     plt.plot(times, entropies)
#     plt.xlabel('Time')
#     plt.ylabel('Entropy')
#     plt.title('Entropy Over Time')
#     plt.grid()
#     plt.show()


# def investigate_initial_conditions_2d():
#     speeds = [1, 5, 10, 20, 30]
#     num_balls = [10, 20, 30, 50, 100]
#     results = []

#     for speed in speeds:
#         for n_balls in num_balls:
#             sim = MultiBallSimulation(b_speed=speed, nrings=n_balls // 10, multi=10)
#             sim.run(100)
#             entropy = sim.entropy_2d()
#             results.append((speed, n_balls, entropy))
#             print(f"Speed: {speed}, Num Balls: {n_balls}, Entropy: {entropy}")

#     speeds, num_balls, entropies = zip(*results)
    
#     fig, ax = plt.subplots()
#     scatter = ax.scatter(speeds, num_balls, c=entropies, cmap='viridis')
#     colorbar = fig.colorbar(scatter, ax=ax)
#     colorbar.set_label('Entropy')
#     ax.set_xlabel('Initial Speed')
#     ax.set_ylabel('Number of Balls')
#     ax.set_title('Entropy for Different Initial Conditions in 2D')
#     plt.grid()
#     plt.show()

#     return results







if __name__ == "__main__":

    # Run task 9 function
    #BALL_POS, BALL_VEL = task9()

    # Run task 10 function
    task10()

    # Run task 11 function
    #task11()

    # Run task 12 function
    #task12()

    # Run task 13 function
    #task13()

    # Run task 14 function
    #task14()

    # Run task 15 function
    #task15()

    #run_simulation(100)

    #investigate_initial_conditions_2d()

    plt.show()
