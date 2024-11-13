import src.material.fluid.basefluid as basefluid
import taichi as ti
import numpy as np
from ..utils import *

@ti.data_oriented
class WCSPH(basefluid.BaseFluid):
    def __init__(self, mesh: trimesh.Trimesh,
                 position=np.array([0.0, 0.0, 0.0]), 
                 gravity=np.array([0.0, -9.8, 0.0]), 
                 viscosity=10, rest_density=1000,
                 time_step=0.0005, fps=60):
        super().__init__(mesh, position, gravity, viscosity, rest_density, time_step, fps)
        
    @ti.func
    def compute_densities_and_pressures(self, p_i, p_j):
        r = self.positions[p_i] - self.positions[p_j]
        r_len = r.norm()
        self.densities[p_i] += self.kernel_func(r_len) * self.mass[p_j]

    @ti.func
    def compute_forces(self, i, j):
        # pressure_force = ti.Vector([0.0, 0.0, 0.0])
        # viscosity_force = ti.Vector([0.0, 0.0, 0.0])
        # surface_tension_force = ti.Vector([0.0, 0.0, 0.0])
        
        r = self.positions[i] - self.positions[j]
        r_len = r.norm()
        nabla_ij = self.kernel_grad(r)
        
        pressure_force = ti.Vector([0.0, 0.0, 0.0])
        if i != j:
            pressure_force = -self.mass[j] * (self.pressures[i] / self.densities[i] ** 2 + self.pressures[j] / self.densities[j] ** 2) * nabla_ij
        v_xy = ti.math.dot(self.velocities[i] - self.velocities[j], r)
        m_ij = (self.mass[i] + self.mass[j]) / 2
        viscosity_force = 2 * 5 * self.viscosity * m_ij / self.densities[j] / (r_len ** 2 + 0.01 * self.h ** 2) * v_xy * nabla_ij / self.rest_density
        surface_tension_force = ti.Vector([0.0, 0.0, 0.0])
        if r_len > self.particle_diameter:
            surface_tension_force = -self.surface_tension / self.densities[i] * self.densities[j] * r * self.kernel_func(r_len)
        else:
            surface_tension_force = -self.surface_tension / self.densities[i] * self.densities[j] * r * self.kernel_func(self.particle_diameter)
        self.forces[i] += pressure_force + viscosity_force + surface_tension_force