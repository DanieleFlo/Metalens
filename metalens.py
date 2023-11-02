# standard python imports
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import math

import tidy3d as td
from tidy3d import web

start_sim = True

# 1 nanometer in units of microns (for conversion)
nm = 1e-3

# free space central wavelength
wavelength = 633 * nm

# desired numerical aperture
NA = 0.2

# shape parameters of metalens unit cell (um) (refer to image above and see paper for details)
W = 200 * nm
H = 1500 * nm  # Altezza del pillars
S = 430 * nm  # Dimensione della cella

# space between bottom PML and substrate (-z)
space_below_sub = 1 * wavelength

# thickness of substrate
thickness_sub = 100 * nm

# side length of entire metalens (um)
side_length = 20

# Number of unit cells in each x and y direction (NxN grid)
N = int(side_length / S)
print(f"for diameter of {side_length:.1f} um, have {N} cells per side")
print(
    f"full metalens has area of {side_length**2:.1f} um^2 and {N*N} total cells")

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

metalens_geometry = []

for x in centers_x:
    for y in centers_y:
        dis = math.sqrt(x**2+y**2)
        phi = (2*math.pi/wavelength) * \
            (math.sqrt(x**2+y**2+f**2)-f) % (2*math.pi)
        alpha = W/(2*math.pi)
        r = W-(phi*alpha)
        # print(dis, length_xy)
        # print('W:', W, '; phi:', phi, '; r', r)
        if r > 0 and dis <= length_xy/2:
            metalens_geometry.append(
                td.Cylinder(
                    axis=2,
                    center=(x, y, center_z),
                    radius=r,
                    length=H,
                )
            )

metalens = td.Structure(
    geometry=td.GeometryGroup(geometries=metalens_geometry), medium=SiO2
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
                medium=SiO2,
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
    center=[0.0, 0.0, 0], size=[0, 0, td.inf], freqs=[f0], name="center",
    colocate=True,
)

# get the fields at a few cross-sectional planes
monitor_xz = td.FieldMonitor(
    center=[0.0, 0.0, 0.0],
    size=[td.inf, 0.0, td.inf],
    freqs=[f0],
    name="xz",
    fields=fields,
    colocate=True,
)

monitor_yz = td.FieldMonitor(
    center=[0.0, 0.0, 0.0],
    size=[0.0, td.inf, td.inf],
    freqs=[f0],
    name="yz",
    fields=fields,
    colocate=True,
)

monitor_xy = td.FieldMonitor(
    center=[0.0, 0.0, center_z + H / 2 + f],
    size=[td.inf, td.inf, 0],
    freqs=[f0],
    name="focal_plane",
    fields=fields,
    colocate=True,
)

# put them into a single list
monitors = [monitor_center, monitor_xz, monitor_yz, monitor_xy]

if start_sim == True:
    sim = td.Simulation(
        size=sim_size,
        #grid_spec=grid_spec,
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
    plt.title("Strutture")
    plt.show()

    # sim.plot_3d()

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
    sim_data.plot_field("xz", "Ex", val="real", f=f0,
                        y=0, ax=ax1, vmin=-15, vmax=15)
    sim_data.plot_field("yz", "Ex", val="real", f=f0,
                        x=0, ax=ax2, vmin=-15, vmax=15)
    sim_data.plot_field("focal_plane", "Ex", val="real", f=f0,
                        z=focal_z, ax=ax3, vmin=-15, vmax=15)
    ax1.set_title("x-z plane")
    ax2.set_title("y-z plane")
    ax3.set_title("x-y (focal) plane")
    plt.title("Parte reale")
    plt.show(block=False)

    # plot absolute value for good measure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    sim_data.plot_field("xz", "Ex", val="abs", f=f0,
                        y=0, ax=ax1, vmin=0, vmax=15)
    sim_data.plot_field("yz", "Ex", val="abs", f=f0,
                        x=0, ax=ax2, vmin=0, vmax=15)
    sim_data.plot_field("focal_plane", "Ex", val="abs", f=f0,
                        z=focal_z, ax=ax3, vmin=0, vmax=15)
    ax1.set_title("x-z plane")
    ax2.set_title("y-z plane")
    ax3.set_title("x-y (focal) plane")
    plt.title("Abs")
    plt.show(block=False)

    # and let's plot the intensites as well
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    sim_data.plot_field("xz", "E", "abs^2", f=f0,
                        y=0, ax=ax1, vmin=0, vmax=3e2)
    sim_data.plot_field("yz", "E", "abs^2", f=f0,
                        x=0, ax=ax2, vmin=0, vmax=3e2)
    sim_data.plot_field("focal_plane", "E", "abs^2", f=f0,
                        z=focal_z, ax=ax3, vmin=0, vmax=3e2)
    ax1.set_title("x-z plane")
    ax2.set_title("y-z plane")
    ax3.set_title("x-y (focal) plane")
    plt.title("Abs^2")
    plt.show()
