# standard python imports
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

import tidy3d as td
from tidy3d import web

# 1 nanometer in units of microns (for conversion)
nm = 1e-3

# free space central wavelength
wavelength = 600 * nm

# desired numerical aperture
NA = 0.5

# shape parameters of metalens unit cell (um) (refer to image above and see paper for details)
W = 85 * nm
L = 410 * nm
H = 600 * nm
S = 430 * nm

# space between bottom PML and substrate (-z)
space_below_sub = 1 * wavelength

# thickness of substrate
thickness_sub = 100 * nm

# side length of entire metalens (um)
side_length = 10

# Number of unit cells in each x and y direction (NxN grid)
N = int(side_length / S)
print(f"for diameter of {side_length:.1f} um, have {N} cells per side")
print(f"full metalens has area of {side_length**2:.1f} um^2 and {N*N} total cells")

# Define material properties at 600 nm
n_TiO2 = 2.40
n_SiO2 = 1.46
air = td.Medium(permittivity=1.0)
SiO2 = td.Medium(permittivity=n_SiO2**2)
TiO2 = td.Medium(permittivity=n_TiO2**2)

# using the wavelength in microns, one can use td.C_0 (um/s) to get frequency in Hz
# wavelength_meters = wavelength * meters
f0 = td.C_0 / wavelength

# Compute the domain size in x, y (note: round down from side_length)
length_xy = N * S

# focal length given diameter and numerical aperture
f = length_xy / 2 / NA * np.sqrt(1 - NA**2)

# Function describing the theoretical best angle of each box at position (x,y).  see paper for details
def theta(x, y):
    return np.pi / wavelength * (f - np.sqrt(x**2 + y**2 + f**2))


# total domain size in z: (space -> substrate -> unit cell -> 1.7 focal lengths)
length_z = space_below_sub + thickness_sub + H + 1.7 * f

# construct simulation size array
sim_size = (length_xy, length_xy, length_z)

# define substrate
substrate = td.Structure(
    geometry=td.Box(
        center=[0, 0, -length_z / 2 + space_below_sub + thickness_sub / 2.0],
        size=[td.inf, td.inf, thickness_sub],
    ),
    medium=SiO2,
)

# define coordinates of each unit cell
centers_x = S * np.arange(N) - length_xy / 2.0 + S / 2.0
centers_y = S * np.arange(N) - length_xy / 2.0 + S / 2.0
center_z = -length_z / 2 + space_below_sub + thickness_sub + H / 2.0

# x, y vertices of box of size (L, W) centered at the origin
vertices_origin = np.array(
    [[+L / 2, +W / 2], [-L / 2, +W / 2], [-L / 2, -W / 2], [+L / 2, -W / 2]]
)


xs, ys = np.meshgrid(centers_x, centers_y, indexing="ij")
xs = xs.flatten()
ys = ys.flatten()

angles = theta(xs, ys)

# 2x2 rotation matrix angle `angle` with respect to x axis
rotation_matrix = np.array(
    [[+np.cos(angles), -np.sin(angles)], [+np.sin(angles), +np.cos(angles)]]
)

# rotate the origin vertices by this angle
vertices_rotated = np.einsum("ij, jkn -> nik", vertices_origin, rotation_matrix)

# shift the rotated vertices to be centered at (xs, ys)
vertices_shifted = vertices_rotated + np.stack([xs, ys], axis=-1)[:, None, :]

metalens_geometry = []
for vertices in vertices_shifted:
    # create a tidy3D PolySlab with these rotated and shifted vertices and thickness `H`
    metalens_geometry.append(
        td.PolySlab(
            vertices=vertices.tolist(),
            slab_bounds=(center_z - H / 2, center_z + H / 2),
            axis=2,
        ),
    )

metalens = td.Structure(
    geometry=td.GeometryGroup(geometries=metalens_geometry), medium=TiO2
)

# steps per unit cell along x and y
grids_per_unit_length = 20

# uniform mesh in x and y
grid_x = td.UniformGrid(dl=S / grids_per_unit_length)
grid_y = td.UniformGrid(dl=S / grids_per_unit_length)

# in z, use an automatic nonuniform mesh with the wavelength being the "unit length"
grid_z = td.AutoGrid(min_steps_per_wvl=grids_per_unit_length)

# we need to supply the wavelength because of the automatic mesh in z
grid_spec = td.GridSpec(
    wavelength=wavelength, grid_x=grid_x, grid_y=grid_y, grid_z=grid_z
)

# put an override box over the pillars to avoid parsing a large amount of structures in the mesher
grid_spec = grid_spec.copy(
    update=dict(
        override_structures=[
            td.Structure(
                geometry=td.Box.from_bounds(
                    rmin=(-td.inf, -td.inf, -length_z / 2 + space_below_sub),
                    rmax=(td.inf, td.inf, center_z + H / 2),
                ),
                medium=TiO2,
            )
        ]
    )
)


# Bandwidth in Hz
fwidth = f0 / 10.0

# time dependence of source
gaussian = td.GaussianPulse(freq0=f0, fwidth=fwidth, phase=0)

source = td.PlaneWave(
    source_time=gaussian,
    size=(td.inf, td.inf, 0),
    center=(0, 0, -length_z / 2 + space_below_sub / 10.0),
    direction="+",
    pol_angle=0,
)

run_time = 50 / fwidth

# To decrease the amount of data stored, only store the E field in 2D monitors
fields = ["Ex", "Ey", "Ez"]

# get fields along x=y=0 axis
monitor_center = td.FieldMonitor(
    center=[0.0, 0.0, 0], size=[0, 0, td.inf], freqs=[f0], name="center"
)

# get the fields at a few cross-sectional planes
monitor_xz = td.FieldMonitor(
    center=[0.0, 0.0, 0.0],
    size=[td.inf, 0.0, td.inf],
    freqs=[f0],
    name="xz",
    fields=fields,
)

monitor_yz = td.FieldMonitor(
    center=[0.0, 0.0, 0.0],
    size=[0.0, td.inf, td.inf],
    freqs=[f0],
    name="yz",
    fields=fields,
)

monitor_xy = td.FieldMonitor(
    center=[0.0, 0.0, center_z + H / 2 + f],
    size=[td.inf, td.inf, 0],
    freqs=[f0],
    name="focal_plane",
    fields=fields,
)

# put them into a single list
monitors = [monitor_center, monitor_xz, monitor_yz, monitor_xy]


sim = td.Simulation(
    size=sim_size,
    grid_spec=grid_spec,
    structures=[substrate, metalens],
    sources=[source],
    monitors=monitors,
    run_time=run_time,
    boundary_spec=td.BoundarySpec(
        x=td.Boundary.absorber(), y=td.Boundary.absorber(), z=td.Boundary.pml()
    ),
)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))

sim.plot(x=0.1, ax=ax1)
sim.plot(y=0.1, ax=ax2)
sim.plot(z=-length_z / 2 + space_below_sub + thickness_sub + H / 2, ax=ax3)
plt.show()


#sim.plot_3d()


job = web.Job(simulation=sim, task_name="metalens", verbose=True)
sim_data = job.run(path="data/simulation_data.hdf5")

print(sim_data.log)

focal_z = center_z + H / 2 + f
data_center_line = sim_data["center"]
I = (
    abs(data_center_line.Ex) ** 2
    + abs(data_center_line.Ey) ** 2
    + abs(data_center_line.Ez) ** 2
)
I.plot()
plt.title("intensity(z)")
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
sim_data.plot_field("xz", "Ex", val="real", f=f0, y=0, ax=ax1, vmin=-15, vmax=15)
sim_data.plot_field("yz", "Ex", val="real", f=f0, x=0, ax=ax2, vmin=-15, vmax=15)
sim_data.plot_field("focal_plane", "Ex", val="real", f=f0, z=focal_z, ax=ax3, vmin=-15, vmax=15)
ax1.set_title("x-z plane")
ax2.set_title("y-z plane")
ax3.set_title("x-y (focal) plane")
plt.show()

# plot absolute value for good measure
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
sim_data.plot_field("xz", "Ex", val="abs", f=f0, y=0, ax=ax1, vmin=0, vmax=15)
sim_data.plot_field("yz", "Ex", val="abs", f=f0, x=0, ax=ax2, vmin=0, vmax=15)
sim_data.plot_field("focal_plane", "Ex", val="abs", f=f0, z=focal_z, ax=ax3, vmin=0, vmax=15)
ax1.set_title("x-z plane")
ax2.set_title("y-z plane")
ax3.set_title("x-y (focal) plane")
plt.show()


# and let's plot the intensites as well
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
sim_data.plot_field("xz", "E", "abs^2", f=f0, y=0, ax=ax1, vmin=0, vmax=3e2)
sim_data.plot_field("yz", "E", "abs^2", f=f0, x=0, ax=ax2, vmin=0, vmax=3e2)
sim_data.plot_field("focal_plane", "E", "abs^2", f=f0, z=focal_z, ax=ax3, vmin=0, vmax=3e2)
ax1.set_title("x-z plane")
ax2.set_title("y-z plane")
ax3.set_title("x-y (focal) plane")
plt.show()

# create a field projection monitor in Cartesian coordinates which records near fields just above the strucure,
# and projects them to points on the focal plane

# number of focal plane sampling points in the x and y directions
num_far = 200

# define the focal plane sample points at which to project fields
xs_far = np.linspace(-sim_size[0] / 2, sim_size[0] / 2, num_far)
ys_far = np.linspace(-sim_size[1] / 2, sim_size[1] / 2, num_far)

pos_monitor_z = -6
proj_distance = monitor_xy.center[2] - pos_monitor_z
monitor_proj = td.FieldProjectionCartesianMonitor(
    center=[0.0, 0.0, pos_monitor_z],
    size=[td.inf, td.inf, 0],
    freqs=[f0],
    name="focal_plane_proj",
    proj_axis=2,  # axis along which to project, in this case z
    proj_distance=proj_distance,  # distance from this monitor to where fields are projected
    x=xs_far,
    y=ys_far,
    custom_origin=[0.0, 0.0, pos_monitor_z],
    far_field_approx=False,  # the distance to the focal plane is comparable to the size of the structure, so
    # turn off the far-field approximation and use an exact Green's function to
    # project the fields
)

# create a simulation as before, but this time there's no need to include the large amount of
# empty space up to the focal plane, and include the projection monitor

# total domain size in z: (space -> substrate -> unit cell -> space)
length_z_new = space_below_sub + thickness_sub + 2 * H + space_below_sub
sim_size = (length_xy, length_xy, length_z_new)
sim_center = (0, 0, -length_z / 2 + length_z_new / 2)
sim_new = td.Simulation(
    size=sim_size,
    center=sim_center,
    grid_spec=grid_spec,
    structures=[substrate, metalens],
    sources=[source],
    monitors=[monitor_proj],
    run_time=run_time,
    boundary_spec=td.BoundarySpec(
        x=td.Boundary.absorber(), y=td.Boundary.absorber(), z=td.Boundary.pml()
    ),
)

# visualize to make sure everything looks okay
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))
sim_new.plot(x=0.1, ax=ax1)
sim_new.plot(y=0.1, ax=ax2)
sim_new.plot(z=-length_z / 2 + space_below_sub + thickness_sub + H / 2, ax=ax3)
plt.show()

job = web.Job(simulation=sim_new, task_name="metalens", verbose=True)
sim_data_new = job.run(path="data/simulation_data_new.hdf5")
print(sim_data_new.log)

# plot the focal plane electric field components recorded by directly placing a monitor in the first simulation,
# which we can use as a reference
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.3, 3.6))
focal_field_data = sim_data["focal_plane"]
Ex = focal_field_data.Ex.sel(f=f0, z=focal_z)
Ey = focal_field_data.Ey.sel(f=f0, z=focal_z)
Ez = focal_field_data.Ez.sel(f=f0, z=focal_z)
im1 = ax1.pcolormesh(Ex.y, Ex.x, np.real(Ex), cmap="RdBu", shading="auto")
im2 = ax2.pcolormesh(Ey.y, Ey.x, np.real(Ey), cmap="RdBu", shading="auto")
im3 = ax3.pcolormesh(Ez.y, Ez.x, np.real(Ez), cmap="RdBu", shading="auto")
fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)
fig.colorbar(im3, ax=ax3)

# now plot the projected fields computed via the second simulation
proj_fields = sim_data_new["focal_plane_proj"].fields_cartesian.sel(
    f=f0, z=monitor_proj.proj_distance
)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.3, 3.6))
im1 = ax1.pcolormesh(
    ys_far, xs_far, np.real(proj_fields.Ex), cmap="RdBu", shading="auto"
)
im2 = ax2.pcolormesh(
    ys_far, xs_far, np.real(proj_fields.Ey), cmap="RdBu", shading="auto"
)
im3 = ax3.pcolormesh(
    ys_far, xs_far, np.real(proj_fields.Ez), cmap="RdBu", shading="auto"
)
fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)
fig.colorbar(im3, ax=ax3)
plt.show()