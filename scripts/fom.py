##
#
#  Assortment of full order models (FOMs).
#
##

# jax imports
import jax.numpy as jnp

##################################################################################
# DOUBLE INTEGRATOR
##################################################################################

class DoubleIntegrator:
    """
    Double Integrator Dynamics Class
    """
    # initialization
    def __init__(self):

        # state and input dimensions
        self.nx = 2
        self.nu = 1

    # dynamics function, continuous time
    @staticmethod
    def f(t, x, u):
        """
        Double Integrator Dynamics
        ẋ = Ax + Bu
        
        Args:
            t: time 
            x: state, shape (2,)
            u: control, shape (1,)
        Returns:
            xdot: derivative, shape (2,)
        """

        # include the parameters here to make pure functions for jitting and vmap
        A = jnp.array([[0.0, 1.0],
                       [0.0, 0.0]]) # (2, 2)
        B = jnp.array([0.0, 1.0])   # (2,)

        # compute the dynamics
        xdot = A @ x + B * u[0]     # (2,)

        return xdot
    
    # feedback controller
    @staticmethod
    def k(t, x):
        """
        Simple PD controller
        u = -Kx
        
        Args:
            t: time 
            x: state, shape (2,)
        Returns:
            u: control, shape (1,)
        """

        # controller gains
        kp = 10.0  
        kd = 1.0   
        
        # compute control input
        p_des = 1.0    # desired position
        v_des = 0.0    # desired velocity
        u = kp * (p_des - x[0]) + kd * (v_des - x[1])  # scalar

        return jnp.array([u])  # make it shape (1,)

##################################################################################
# PENDULUM
##################################################################################

class Pendulum:
    """
    Pendulum Dynamics Class
    """
    # initialization
    def __init__(self):

        # state and input dimensions
        self.nx = 2
        self.nu = 1

    # dynamics function, continuous time
    @staticmethod
    def f(t, x, u):
        """
        Pendulum Dynamics (underactuated robotics Ch 2.1)
        m l² θ̈ + b θ̇ + m g l sin(θ) = τ

        Args:
            t: time 
            x: state, shape (2,)
            u: control, shape (1,)
        Returns:
            xdot: derivative, shape (2,)
        """

        # include the parameters here to make pure functions for jitting and vmap
        L = 1.0   # length of the pendulum
        m = 1.0   # mass of the pendulum
        g = 9.81  # acceleration due to gravity
        b = 0.1   # damping coefficient        

        # extract state variables
        theta = x[0]      # angle
        theta_dot = x[1]  # angular velocity
        tau = u[0]        # control input

        # dynamics
        theta_ddot = (tau - m * g * L * jnp.sin(theta) - b * theta_dot) / (m * L**2)

        # build the dynamics vector
        x_dot = jnp.array([theta_dot, theta_ddot])

        return x_dot
        
    # feedback controller
    @staticmethod
    def k(t, x):
        """
        Simple PD controller
        """

        # controller gains
        kp = 10.0
        kd = 1.0

        # extract state variables
        theta = x[0]      # theta (angle)
        theta_dot = x[1]  # theta_dot (angular velocity)
    
        # compute the control input
        theta_des = 3.14     # desired angle (radians)
        theta_dot_des = 0.0  # desired angular velocity
        u = kp * (theta_des - theta) + kd * (theta_dot_des - theta_dot)   # scalar

        return jnp.array([u])  # make it shape (1,)


##################################################################################
# VAN DER POL OSCILLATOR
##################################################################################

class VanDerPol:
    """
    Van der Pol Oscillator Dynamics Class
    """
    # initialization
    def __init__(self, mu=1.0):

        # state and input dimensions
        self.nx = 2
        self.nu = 0

    # dynamics function, continuous time
    def f(self, t, x, u):
        """
        Van der Pol Oscillator Dynamics (μ > 0 determines nonlinearity and the strength of the damping)
        ẋ₁ = x₂
        ẋ₂ = μ (1 - x₁²) x₂ - x₁
        
        Args:
            t: time 
            x: state, shape (2,)
            u: control, shape (0,) or None
        Returns:
            xdot: derivative, shape (2,)
        """

        # include the parameters here to make pure functions for jitting and vmap
        mu = 1.0

        # extract state variables
        x1 = x[0]
        x2 = x[1]

        # dynamics
        x1_dot = x2
        x2_dot = mu * (1 - x1**2) * x2 - x1

        # build the dynamics vector
        xdot = jnp.array([x1_dot, x2_dot])

        return xdot
    
    # no control input for this system
    def k(self, t, x):
        return jnp.array([])  # shape (0,)