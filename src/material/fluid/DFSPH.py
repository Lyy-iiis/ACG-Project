import taichi as ti
import numpy as np
from ..utils import *
from src.material.fluid.basefluid import BaseFluid

@ti.data_oriented
class DFSPH(BaseFluid):
    def __init__(self, mesh: trimesh.Trimesh,
                 position=np.array([0.0, 0.0, 0.0]), 
                 gravity=np.array([0.0, -9.8, 0.0]), 
                 viscosity=10, rest_density=1000,
                 time_step=5e-4, fps=60):
        super().__init__(mesh, position, gravity, viscosity, rest_density, time_step, fps)
    
        self.m_max_iterations_v = 1000
        self.m_max_iterations = 1000

        self.m_eps = 1e-5

        self.max_error_V = 0.001
        self.max_error = 0.0001
    
    @ti.func
    def compute_non_pressure_forces(self, i, j):
        r = self.positions[i] - self.positions[j]
        r_len = r.norm()
        nabla_ij = self.kernel_grad(r)
        
        v_xy = ti.math.dot(self.velocities[i] - self.velocities[j], r)
        m_ij = (self.mass[i] + self.mass[j]) / 2
        viscosity_force = 2 * 5 * self.viscosity * m_ij / self.densities[j] / (r_len ** 2 + 0.01 * self.h ** 2) * v_xy * nabla_ij / self.rest_density
        surface_tension_force = ti.Vector([0.0, 0.0, 0.0])
        if r_len > self.particle_diameter:
            surface_tension_force = -self.surface_tension / self.densities[i] * self.densities[j] * r * self.kernel_func(r_len)
        else:
            surface_tension_force = -self.surface_tension / self.densities[i] * self.densities[j] * r * self.kernel_func(self.particle_diameter)
        self.forces[i] += viscosity_force + surface_tension_force
    
    @ti.kernel
    def update_velocity(self):
        for i in range(self.num_particles):
            self.velocities[i] += self.time_step * self.forces[i] / self.mass[i]
        avg_velocity = ti.Vector([0.0, 0.0, 0.0])
        for i in range(self.num_particles):
            avg_velocity += self.velocities[i]
        avg_velocity /= self.num_particles
        print(avg_velocity)
    
    @ti.kernel
    def update_position(self):
        for i in range(self.num_particles):
            self.positions[i] += self.time_step * self.velocities[i]