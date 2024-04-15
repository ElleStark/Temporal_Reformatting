# FlowField class
# Elle Stark May 2024

class FlowField:

    def __init__(self, xmesh, ymesh, u_data, v_data, xmesh_uv, ymesh_uv, dt_uv, flipuv=True):
        super().__init__()

        self.x = xmesh
        self.y = ymesh
        self.u_data = u_data
        self.v_data = v_data
        self.xmesh_uv = xmesh_uv
        self.ymesh_uv = ymesh_uv
        self.dt_uv = dt_uv

    def vfield(self, time, y):
        """
        Calculates velocity field based on interpolation from existing data.
        :param y: array of particle locations where y[0] is array of x locations and y[1] is array of y locations
        :param time: scalar value for time
        :return: array of u and v, where u is size x by y ndarray of horizontal velocity magnitudes,
        and v is size x by y ndarray of vertical velocity magnitudes.
        """
        # Convert from time to frame
        frame = int(time / self.dt_uv)

        # axes must be in ascending order, so need to flip y-axis, which also means flipping u and v upside-down
        ymesh_vec = np.flipud(self.ymesh_uv)[:, 0]
        xmesh_vec = self.xmesh_uv[0, :]

        # Set up interpolation functions
        # can use cubic interpolation for continuity of the between the segments (improve smoothness)
        # set bounds_error=False to allow particles to go outside the domain by extrapolation
        if flipuv: 
            u_matrix = np.squeeze(np.flipud(self.u_data[:, :, frame]))
            v_matrix = np.squeeze(np.flipud(self.v_data[:, :, frame]))
        else: 
            u_matrix = np.squeeze(self.u_data[:, :, frame])
            v_matrix = np.squeeze(self.v_data[:, :, frame])

        u_interp = RegularGridInterpolator((ymesh_vec, xmesh_vec), u_matrix,
                                           method='linear', bounds_error=False, fill_value=None)
        v_interp = RegularGridInterpolator((ymesh_vec, xmesh_vec), np.squeeze(np.flipud(self.v_data[:, :, frame])),
                                           method='linear', bounds_error=False, fill_value=None)

        # Interpolate u and v values at desired x (y[0]) and y (y[1]) points
        u = u_interp((y[1], y[0]))
        v = v_interp((y[1], y[0]))

        vfield = np.array([u, v])

        return vfield
    
    def compute_vfields(self, t):
        """
        Computes spatial velocity field for list of desired times
        :param t: ndarray of time values at which velocity field will be calculated
        :return: dictionary of velocity fields, one for each time value.
                 Each velocity field is a list of 4 ndarrays: [x, y, u, v].
        """
        vfields = []

        # Loop through time, assigning velocity field [x, y, u, v] for each t
        for time in t:
            vfield = self.vfield(time, [self.x, self.y])
            # need to extract u and v from vfield array
            u = vfield[0]
            v = vfield[1]
            vfield = [self.x, self.y, u, v]
            vfields.append(vfield)
        vfield_dict = dict(zip(t, vfields))

        self.velocity_fields = vfield_dict

