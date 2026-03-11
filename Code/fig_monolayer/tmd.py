"""
Here I try to make a 3d version os the TMD to highlight the hoppings considered in the tight-binding model.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource

""" Light source """
az = 60   # azimuth (degrees)
alt = 30   # elevation (degrees)
t1 = 0.5
t2 = 1-t1
#gamma_sphere = 0.4

az_rad = np.deg2rad(az)
alt_rad = np.deg2rad(alt)
light_dir = np.array([
    np.cos(alt_rad) * np.cos(az_rad),
    np.cos(alt_rad) * np.sin(az_rad),
    np.sin(alt_rad)
])
light_dir /= np.linalg.norm(light_dir)
lightsource = LightSource(azdeg=az,altdeg=alt)

base_color_sphere = np.array([0.2, 0.5, 0.9])  # blue-ish
base_color_ovoid = np.array([0.8, 0.5, 0.9])  # blue-ish
base_color_cube = np.array([238, 165, 71])/256  # uniform material color
ptsSurface = 30

def build_sphere(center,radius,ax,pts=10):
    # Sphere parameters
    u = np.linspace(0, 2 * np.pi, pts)
    v = np.linspace(0, np.pi, pts)

    x = center[0] + radius*np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius*np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius*np.outer(np.ones_like(u), np.cos(v))

    #normals = np.dstack((x, y, z))
    #intensity = np.clip(np.dot(normals, light_dir), 0, 1)
    #intensity = t1 + t2 * intensity**gamma_sphere  # ambient term

    #sphere_rgb = intensity[..., None] * base_color_sphere
    # Plot surface
    ax.plot_surface(
        x, y, z,
        #facecolors=sphere_rgb,
        color=base_color_sphere,
        lightsource=lightsource,
        linewidth=0.5,
        antialiased=True
    )
def build_ovoid(center,z,el,ax,pts=10):
    """ COnstruct ovoidal shape which encompasses the two cubes
    """
    u = np.linspace(0, 2 * np.pi, pts)
    v = np.linspace(0, np.pi, pts)

    radius = el
    radiusz = z
    x = center[0] + radius*np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius*np.outer(np.sin(u), np.sin(v))
    z = center[2] + radiusz*np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(
        x, y, z,
        color=base_color_ovoid,
        lightsource=lightsource,
        linewidth=0.,
        alpha=0.3,
        antialiased=True
    )
def compute_normal(face,vertices):
    """ For cube plotting. """
    v1 = vertices[face[1]] - vertices[face[0]]
    v2 = vertices[face[2]] - vertices[face[0]]
    n = np.cross(v1, v2)
    return n / np.linalg.norm(n)
def build_cube(center,edge_len,ax):
    # ----- Cube geometry -----
    # Cube vertices
    vertices = np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1]
    ]) / 2 * edge_len + center

    # Define faces by vertex indices
    faces = [
        [0,1,2,3],  # bottom
        [4,5,6,7],  # top
        [0,1,5,4],  # front
        [2,3,7,6],  # back
        [1,2,6,5],  # right
        [0,3,7,4]   # left
    ]

    face_colors = []
    poly3d = []
    for face in faces:
        pts = vertices[face]
        poly3d.append(pts)
        normal = compute_normal(face,vertices)
        # Lambertian intensity
        intensity = max(0, np.dot(normal, light_dir))
        # Avoid completely black faces (ambient term)
        intensity = t1 + t2 * intensity
        face_colors.append(base_color_cube * intensity)
    # Plotting
    collection = Poly3DCollection(
        poly3d,
        facecolors=face_colors,
        edgecolors='black',
        linewidths=0.5
    )
    ax.add_collection3d(collection)

""" Plotting """
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

a1 = np.array([1,0,0])              # Lattice vectors -> lattice contanst 1
a2 = np.array([1/2,np.sqrt(3)/2,0]) # Lattice vectopr 2
d1 = np.array([0,1/np.sqrt(3),0])     # Vector of nn
z1 = np.array([0,0,0.5])               # Vertical displacement
radius = 0.15
edge_len = 0.12
centers = np.array([
    np.zeros(3),
    d1,
    a2,
    a1+d1,
    a1,
    a1-a2+d1
])
# Plot cubes and spheres
for i in range(0,6,2):
    # Cubes -> X2
    build_cube(centers[i]-z1/2,edge_len,ax)
    build_cube(centers[i]+z1/2,edge_len,ax)
    build_ovoid(centers[i],z1[-1],edge_len,ax,pts=ptsSurface)
    # Spheres -> MS
    build_sphere(centers[i+1],radius,ax,pts=ptsSurface)
# Hexagonal edges -> unit cell
for i in range(0,6):
    ax.plot(
        [centers[i][0],centers[(i+1)%6][0]],
        [centers[i][1],centers[(i+1)%6][1]],
        [centers[i][2],centers[(i+1)%6][2]],
        color='k',
        lw=0.8,
        ls=(0,(5,5))
    )
if 0:
    # Hopping terms
    i = 3
    ax.plot(
        [centers[i][0],centers[(i+2)%6][0]],
        [centers[i][1],centers[(i+2)%6][1]],
        [centers[i][2],centers[(i+2)%6][2]],
        color='red',
        lw=3,
    )

    i = 2
    ax.plot(
        [centers[i][0],centers[(i+1)%6][0]],
        [centers[i][1],centers[(i+1)%6][1]],
        [centers[i][2],centers[(i+1)%6][2]],
        color='purple',
        lw=3,
    )

    i = 4
    ax.plot(
        [centers[i][0],centers[(i+2)%6][0]],
        [centers[i][1],centers[(i+2)%6][1]],
        [centers[i][2],centers[(i+2)%6][2]],
        color='green',
        lw=3,
    )

    i = 3
    ax.plot(
        [centers[i][0],centers[(i+3)%6][0]],
        [centers[i][1],centers[(i+3)%6][1]],
        [centers[i][2],centers[(i+3)%6][2]],
        color='firebrick',
        lw=3,
    )

""" Set same scale on axes so that spheres are spheres """
xmin,xmax = ax.get_xlim()
ymin,ymax = ax.get_ylim()
zmin,zmax = ax.get_zlim()
max_range = np.array([
    xmax-xmin,
    ymax-ymin,
    zmax-zmin,
]).max() / 2

mid_x = (xmax+xmin) * 0.5
mid_y = (ymax+ymin) * 0.5
mid_z = (zmax+zmin) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_box_aspect([1,1,1])

ax.view_init(elev=30, azim=45)
plt.show()



























