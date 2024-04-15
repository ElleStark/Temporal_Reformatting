# Utility functions for two-particle simulations for temporal reformatting study
# Elle Stark May 2024

def improvedEuler_singlestep(u_field, dt, t0, y0):
        """
        Single step of 2nd-order improved Euler integration. vfield must be a function that returns an array of [u, v] values
        :param u_field: nd_array of velocity data (time, x position, y position)
        :param dt: scalar value of desired time step
        :param t0: scalar start time for integration
        :param y0: nd_array starting position of particles, can be matrix of x, y positions
        :return: nd_array of final position of particles
        """
        # get the slopes at the initial and end points
        f1 = u_field(t0, y0)
        f2 = u_field(t0 + dt, y0 + dt * f1)
        y_out = y0 + dt / 2 * (f1 + f2)

        return y_out

def rk4singlestep(u_field, dt, t0, y0):
    """
    Single step of 4th-order Runge-Kutta integration. Use instead of scipy.integrate.solve_ivp to allow for
    vectorized computation of bundle of initial conditions. Reference: https://www.youtube.com/watch?v=LRF4dGP4xeo
    Note that u_field must be a function that returns an array of [u, v] values
    :param u_field: nd_array of velocity data (time, x position, y position)
    :param dt: scalar value of desired time step
    :param t0: start time for integration
    :param y0: starting position of particles
    :return: final position of particles
    """
    # RK4 first computes velocity at full steps and partial steps
    f1 = u_field(t0, y0)
    f2 = u_field(t0 + dt / 2, y0 + (dt / 2) * f1)
    f3 = u_field(t0 + dt / 2, y0 + (dt / 2) * f2)
    f4 = u_field(t0 + dt, y0 + dt * f3)
    # RK4 then takes a weighted average to move the particle
    y_out = y0 + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)
    return y_out