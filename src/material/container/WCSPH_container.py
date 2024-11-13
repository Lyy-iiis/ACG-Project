from src.material.fluid.basefluid import *
from src.material.rigid import *
from src.material.container.base_container import *
import os
from src.material.geometry import *

@ti.data_oriented
class WCSPHContainer(Container):
    def __init__(self, width, height, depth, fluid, rigid):
        super().__init__(width, height, depth, fluid, rigid)
    
    @ti.func
    def update_rigid_particle_volume(self):
        for i in range(self.rigid_num_particles):
            pos_i = self.rigid_positions[i]
            self.rigid_volumes[i] = 0.0
            for j in range(self.neighbour_num[i+self.fluid.num_particles]):
                p_j = self.neighbour[i+self.fluid.num_particles,j]
                if not self.is_fluid[p_j]:
                    pos_j = self.rigid_positions[p_j-self.fluid.num_particles]
                    R = pos_i - pos_j
                    R_mod = R.norm()
                    self.rigid_volumes[i] += self.fluid.kernel_func(R_mod)
            self.rigid_volumes[i] = 1.0 / self.rigid_volumes[i]
            self.rigid_masses[i] = self.rigid_volumes[i] * self.fluid.rest_density
            
    @ti.func
    def compute_densities_and_pressures(self):
        for i in range(self.fluid.num_particles):
            self.fluid.densities[i] = 0.0
            self.fluid.pressures[i] = 0.0
            for j in range(self.neighbour_num[i]):
                p_j = self.neighbour[i,j]
                if self.is_fluid[p_j]:
                    self.fluid.compute_densities_and_pressures(i, p_j)
                else:
                    pos_i = self.fluid.positions[i]
                    pos_j = self.rigid_positions[p_j-self.fluid.num_particles]
                    R = pos_i - pos_j
                    R_mod = R.norm()
                    self.fluid.densities[i] += self.fluid.kernel_func(R_mod) * self.rigid_masses[p_j-self.fluid.num_particles]
                    
            self.fluid.densities[i] = ti.max(self.fluid.densities[i], self.fluid.rest_density)
            self.fluid.pressures[i] = self.fluid.stiffness * ((self.fluid.densities[i] / self.fluid.rest_density) ** 7 - 1)

        self.fluid.avg_density[None] = 0.0
        for i in range(self.fluid.num_particles):
            self.fluid.avg_density[None] += self.fluid.densities[i]
        self.fluid.avg_density[None] /= self.fluid.num_particles
    
    @ti.func
    def compute_forces_rigid(self, i, p_j):
        m_ij = self.rigid_masses[p_j-self.fluid.num_particles]
        v_xy = self.fluid.velocities[i] - self.rigid_velocities[p_j-self.fluid.num_particles]
        R = self.fluid.positions[i] - self.rigid_positions[p_j-self.fluid.num_particles]
        v_xy = ti.math.dot(v_xy, R)
        nabla_ij = self.fluid.kernel_grad(R)
        
        viscosity_force = 2 * 5 * self.fluid.viscosity * m_ij / self.fluid.densities[i] / (R.norm() ** 2 + 0.01 * self.fluid.h ** 2) * v_xy * nabla_ij / self.fluid.rest_density
        pressure_force = - m_ij * self.fluid.pressures[i] / (self.fluid.densities[i] ** 2) * nabla_ij
        force = pressure_force + viscosity_force
        self.fluid.forces[i] += force
        
        force_j = - force * self.fluid.mass[i] / self.rigid_masses[p_j-self.fluid.num_particles]
        pos_j = self.rigid_positions[p_j-self.fluid.num_particles]
        self.rigid.apply_internal_force(force_j, pos_j)
        
    @ti.func
    def compute_forces(self):
        for i in range(self.fluid.num_particles):
            self.fluid.forces[i] = ti.Vector([0.0, 0.0, 0.0])
            for j in range(self.neighbour_num[i]):
                p_j = self.neighbour[i,j]
                if p_j < self.fluid.num_particles:
                    self.fluid.compute_forces(i, p_j)
                else:
                    self.compute_forces_rigid(i, p_j)
                    
            self.fluid.forces[i] += self.fluid.gravity[None]
            self.fluid.forces[i] *= self.fluid.mass[i]
        
        # self.rigid.force[None] += self.rigid.mass * self.fluid.gravity[None]
    
    @ti.kernel
    def update(self):
        self.empty_grid()
        self.update_grid()
        self.update_neighbour()
        self.update_rigid_particle_volume()
        self.compute_densities_and_pressures()
        self.compute_forces()
        self.fluid.update_particles()
        self.enforce_domain_boundary()
        
    def step(self):
        if self.rigid is not None:
            self.get_rigid_pos()
        self.update()
        # self.rigid.update(self.fluid.time_step)