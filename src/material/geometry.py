import trimesh

def Ball(radius = 1, center = [0,0,0], resolution = 100):
    return trimesh.creation.icosphere(radius = radius, center = center, subdivisions = 2)

def Box(extents = [1,1,1], center = [0,0,0]):
    return trimesh.creation.box(extents = extents, center = center)

def Cylinder(radius = 1, height = 1, center = [0,0,0], resolution = 100):
    return trimesh.creation.cylinder(radius = radius, height = height, center = center, resolution = resolution)

def Plane(size = [1,1], center = [0,0,0]):
    return trimesh.creation.box(extents = [size[0], 0.001, size[1]], center = center)

def Sphere(radius = 1, center = [0,0,0], resolution = 100):
    return trimesh.creation.uv_sphere(radius = radius, center = center, resolution = resolution)

def Cone(radius = 1, height = 1, center = [0,0,0], resolution = 100):
    return trimesh.creation.cone(radius = radius, height = height, center = center, resolution = resolution)

def Torus(inner_radius = 0.5, outer_radius = 1, center = [0,0,0], resolution = 100):
    return trimesh.creation.torus(inner_radius = inner_radius, outer_radius = outer_radius, center = center, resolution = resolution)