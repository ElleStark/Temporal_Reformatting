# Class for odor source objects
# Elle Stark May 2024

class OdorSource:

    def __init__(self, tau, osrc_loc, D_osrc) -> None:
        self.tau = tau  # scalar time spacing of seeded particles
        self.osrc_loc = osrc_loc  # [x, y] location of particle release point
        self.D_osrc = D_osrc  # scalar diffusivity of odor
        

