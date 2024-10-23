import taichi as ti
import numpy as np
from .utils import *

@ti.data_oriented
class Fluid:
    def __init__(self, num_particles, domain_size, particle_radius,
                 mesh=None, position=np.array([0.0, 0.0, 0.0]),
                 gravity=np.array([0.0, -9.8, 0.0]), 
                 viscosity=0.1, rest_density=1000.0, 
                 time_step=0.01):
        self.num_particles = num_particles
        self.domain_size = domain_size
        self.particle_radius = particle_radius
        self.gravity = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.gravity[None] = gravity
        self.viscosity = viscosity
        self.rest_density = rest_density
        self.time_step = time_step

        self.mesh = mesh
        self.mesh.vertices += position
        self.original_positions = ti.Vector(position)
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
        self.densities = ti.field(dtype=ti.f32, shape=num_particles)
        self.pressures = ti.field(dtype=ti.f32, shape=num_particles)
        self.forces = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)

        if mesh is not None:
            min_x, min_y, min_z = np.min(mesh.vertices, axis=0)
            max_x, max_y, max_z = np.max(mesh.vertices, axis=0)
            self.domain_size = [[min_x, min_y, min_z], [max_x, max_y, max_z]]
            for i in range(self.num_particles):
                while True:
                    pos = np.array([[
                        np.random.rand() * (self.domain_size[1][0] - self.domain_size[0][0]) + self.domain_size[0][0],
                        np.random.rand() * (self.domain_size[1][1] - self.domain_size[0][1]) + self.domain_size[0][1],
                        np.random.rand() * (self.domain_size[1][2] - self.domain_size[0][2]) + self.domain_size[0][2]
                    ]])
                    if self.mesh.contains(pos):
                        self.positions[i] = ti.Vector(pos[0])
                        break
        else:
            self.init_random_particles()
        self.init_velocity_and_density()
            
    @ti.kernel
    def init_random_particles(self):
        for i in range(self.num_particles):
            self.positions[i] = ti.Vector([
                ti.random() * self.domain_size[0],
                ti.random() * self.domain_size[1],
                ti.random() * self.domain_size[2]
            ])
            self.velocities[i] = ti.Vector([0.0, 0.0, 0.0])
            self.densities[i] = self.rest_density
            self.pressures[i] = 0.0
            self.forces[i] = ti.Vector([0.0, 0.0, 0.0])
        
    @ti.kernel
    def init_velocity_and_density(self):
        for i in range(self.num_particles):
            self.velocities[i] = ti.Vector([0.0, 0.0, 0.0])
            self.densities[i] = self.rest_density
            self.pressures[i] = 0.0
            self.forces[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def compute_densities_and_pressures(self):
        for i in range(self.num_particles):
            density = 0.0
            for j in range(self.num_particles):
                r = self.positions[i] - self.positions[j]
                r_len = r.norm()
                if r_len < self.particle_radius:
                    density += (self.particle_radius - r_len) ** 3
            self.densities[i] = density
            self.pressures[i] = self.viscosity * (self.densities[i] - self.rest_density)

    @ti.kernel
    def compute_forces(self):
        for i in range(self.num_particles):
            force = self.gravity[None] * self.densities[i]
            for j in range(self.num_particles):
                if i != j:
                    r = self.positions[i] - self.positions[j]
                    r_len = r.norm()
                    if r_len < self.particle_radius:
                        pressure_force = -self.pressures[i] * (self.particle_radius - r_len) * r.normalized()
                        force += pressure_force
            self.forces[i] = force

    @ti.kernel
    def update_particles(self):
        for i in range(self.num_particles):
            self.velocities[i] += self.forces[i] * self.time_step / self.densities[i]
            self.positions[i] += self.velocities[i] * self.time_step

    def step(self):
        self.compute_densities_and_pressures()
        self.compute_forces()
        self.update_particles()
        
    def positions_to_ply(self, output_path):
        positions = self.positions.to_numpy()
        write_ply(positions, output_path)