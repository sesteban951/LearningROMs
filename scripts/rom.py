##
#
# Assortment of reduced order models (ROMs).
#
##

import jax.numpy as jnp         # standard jax numpy

##################################################################################
# DOUBLE INTEGRATOR
##################################################################################

# general ODE solver class
class DoubleIntegrator:
    """
        Double Integrator Dynamics Class
    """
    # initialization
    def __init__(self):

        # name
        self.name = "DoubleIntegrator"

        # state and input dimensions
        self.nx = 2
        self.nu = 1

    # dynamics function, continuous time
    @staticmethod
    def f(x, u):
        """
            Double Integrator Dynamics
            ẋ = Ax + Bu
            
            Args:
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
            x: state, shape (2,)
            t: time (optional, for future extensions)
        Returns:
            u: control, shape (1,)
        """
        # controller gains
        kp = 10.0  
        kd = 1.0   
        K = jnp.array([kp, kd])  # (2,) gains for [position, velocity]
        
        # compute control input
        u = -jnp.dot(K, x)     # scalar

        return jnp.array([u])  # make it shape (1,)

##################################################################################
# PENDULUM
##################################################################################