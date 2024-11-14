from src.material.fluid import basefluid, DFSPH
from src.material.rigid import *
from src.material.container.base_container import *
import os
from src.material.geometry import *

@ti.data_oriented
class DFSPHContainer(Container):
    def __init__(self, width, height, depth, fluid: DFSPH.DFSPH, rigid):
        super().__init__(width, height, depth, fluid, rigid)
        self.fluid = fluid
        self.alpha = ti.field(dtype=ti.f32, shape=self.fluid.num_particles)
        self.kappa = ti.field(dtype=ti.f32, shape=self.fluid.num_particles)
        self.kappa_v = ti.field(dtype=ti.f32, shape=self.fluid.num_particles)
        self.rho_star = ti.field(dtype=ti.f32, shape=self.fluid.num_particles)
        self.rho_derivative = ti.field(dtype=ti.f32, shape=self.fluid.num_particles)
        
    @ti.kernel
    def compute_density(self):
        for i in range(self.fluid.num_particles):
            self.fluid.densities[i] = 0.0
            for j in range(self.neighbour_num[i]):
                p_j = self.neighbour[i, j]
                if self.is_fluid[p_j]:
                    r = self.fluid.positions[i] - self.fluid.positions[p_j]
                    r_len = r.norm()
                    self.fluid.densities[i] += self.fluid.mass[p_j] * self.fluid.kernel_func(r_len)
                else:
                    pos_i = self.fluid.positions[i]
                    pos_j = self.rigid_positions[p_j - self.fluid.num_particles]
                    r = pos_i - pos_j
                    r_len = r.norm()
                    self.fluid.densities[i] += self.rigid_masses[p_j - self.fluid.num_particles] * self.fluid.kernel_func(r_len)
                    
        # for i in range(self.fluid.num_particles):
        #     self.fluid.densities[i] = ti.max(self.fluid.densities[i], self.fluid.rest_density)
            
        avg_density = 0.0
        for i in range(self.fluid.num_particles):
            avg_density += self.fluid.densities[i]
        avg_density /= self.fluid.num_particles
        print("avg_density: ", avg_density)
        
        
    @ti.kernel
    def compute_alpha(self):
        """
        compute alpha for each particle
        """
        for i in range(self.fluid.num_particles):
            sum_grad_p_k = 0.0
            grad_p_i = ti.Vector([0.0, 0.0, 0.0])
            for j in range(self.neighbour_num[i]):
                p_j = self.neighbour[i, j]
                if self.is_fluid[p_j]:
                    r = self.fluid.positions[i] - self.fluid.positions[p_j]
                    grad_p_j = self.fluid.mass[p_j] * self.fluid.kernel_grad(r)
                    sum_grad_p_k += grad_p_j.norm_sqr()
                    grad_p_i += grad_p_j
                else:
                    r = self.fluid.positions[i] - self.rigid_positions[p_j - self.fluid.num_particles]
                    grad_p_i += self.rigid_masses[p_j - self.fluid.num_particles] * self.fluid.kernel_grad(r)

            sum_grad_p_k += grad_p_i.norm_sqr()
            factor = 0.0
            if sum_grad_p_k > 1e-5:
                factor = 1.0 / sum_grad_p_k
            else:
                factor = 0.0
            self.alpha[i] = factor * self.fluid.densities[i] 
            
    @ti.kernel
    def compute_density_derivative(self):
        """
        compute (D rho / Dt) / rho_0 for each particle
        """
        for i in range(self.fluid.num_particles):
            if self.neighbour_num[i] < 20:
                self.rho_derivative[i] = 0.0
                continue
            ret = 0.0
            for j in range(self.neighbour_num[i]):
                p_j = self.neighbour[i, j]
                if self.is_fluid[p_j]:
                    r = self.fluid.positions[i] - self.fluid.positions[p_j]
                    v_i = self.fluid.velocities[i]
                    v_j = self.fluid.velocities[p_j]
                    ret += self.fluid.particle_volume[p_j] * ti.math.dot(v_i - v_j, self.fluid.kernel_grad(r))
                else:
                    r = self.fluid.positions[i] - self.rigid_positions[p_j - self.fluid.num_particles]
                    v_i = self.fluid.velocities[i]
                    v_j = self.rigid_velocities[p_j - self.fluid.num_particles]
                    ret += self.rigid_volumes[p_j - self.fluid.num_particles] * ti.math.dot(v_i - v_j, self.fluid.kernel_grad(r))
            self.rho_derivative[i] = ti.max(ret, 0.0)
    
    @ti.kernel
    def compute_rho_star(self):
        """
        compute rho^* / rho_0 for each particle
        """
        for i in range(self.fluid.num_particles):
            delta = 0.0
            v_i = self.fluid.velocities[i]
            pos_i = self.fluid.positions[i]
            for j in range(self.neighbour_num[i]):
                p_j = self.neighbour[i, j]
                pos_j = ti.Vector([0.0, 0.0, 0.0])
                v_j = ti.Vector([0.0, 0.0, 0.0])
                if self.is_fluid[p_j]:
                    v_j = self.fluid.velocities[p_j]
                    pos_j = self.fluid.positions[p_j]
                else:
                    v_j = self.rigid_velocities[p_j - self.fluid.num_particles]
                    pos_j = self.rigid_positions[p_j - self.fluid.num_particles]
                r = pos_i - pos_j
                delta += self.fluid.particle_volume[i] * ti.math.dot(v_i - v_j, self.fluid.kernel_grad(r))
            # print(delta)
            self.rho_star[i] = self.fluid.densities[i] / self.fluid.rest_density + self.fluid.time_step * delta
            self.rho_star[i] = ti.max(self.rho_star[i], 1.0)

        rho_star_avg = 0.0
        avg_rho = 0.0
        for i in range(self.fluid.num_particles):
            rho_star_avg += self.rho_star[i]
            avg_rho += self.fluid.densities[i]
        avg_rho /= self.fluid.num_particles
        rho_star_avg /= self.fluid.num_particles
        print("rho_star_avg: ", rho_star_avg)
        print("avg_rho: ", avg_rho)
    
    @ti.kernel
    def compute_kappa_v(self):
        """
        compute kappa_v = alpha * (D rho / Dt) for each particle
        """
        for i in range(self.fluid.num_particles):
            self.kappa_v[i] = self.rho_derivative[i] * self.alpha[i]
            # print("kappa_v", self.kappa_v)
        
    def correct_divergence(self):
        iteration = 0
        self.compute_density_derivative()
        average_density_derivative_error = 0.0
        
        while iteration < 1 or iteration < self.fluid.m_max_iterations_v:
            self.compute_kappa_v()
            self.correct_divergence_step()
            self.compute_density_derivative()
            average_density_derivative_error = self.compute_density_derivative_error()
            eta = self.fluid.max_error_V * self.fluid.rest_density / self.fluid.time_step
            iteration += 1
            # print(average_density_derivative_error, eta)
            if average_density_derivative_error <= eta:
                break
    
        print(f"DFSPH - iteration V: {iteration} Avg density derivative err: {average_density_derivative_error}")
    
    @ti.kernel
    def correct_divergence_step(self):
        for i in range(self.fluid.num_particles):
            k_i = self.kappa_v[i]
            pos_i = self.fluid.positions[i]
            den_i = self.fluid.densities[i]
            ret = ti.Vector([0.0, 0.0, 0.0])
            for j in range(self.neighbour_num[i]):
                p_j = self.neighbour[i, j]
                if self.is_fluid[p_j]:
                    k_j = self.kappa_v[p_j]
                    k_sum = k_i + k_j
                    if ti.abs(k_sum) > self.fluid.m_eps * self.fluid.time_step:
                        pos_j = self.fluid.positions[p_j]
                        den_j = self.fluid.densities[p_j]
                        r = pos_i - pos_j
                        grad_p_j = self.fluid.mass[j] * self.fluid.kernel_grad(r)
                        ret -= grad_p_j * (k_i / den_i + k_j / den_j)
                else:
                    k_sum = k_i
                    if ti.abs(k_sum) > self.fluid.m_eps * self.fluid.time_step * self.fluid.rest_density:
                        pos_j =  self.rigid_positions[p_j - self.fluid.num_particles]
                        r = pos_i - pos_j
                        grad_p_j = self.rigid_masses[p_j - self.fluid.num_particles] * self.fluid.kernel_grad(r)
                        ret -= grad_p_j * (k_i / den_i)
                        force_j = grad_p_j * (k_i / den_i) * self.fluid.mass[i] / self.fluid.time_step
                        self.rigid.apply_internal_force(force_j, pos_j)
            self.fluid.velocities[i] += ret

    @ti.kernel
    def compute_density_derivative_error(self) -> float:
        """
        compute average of (D rho / Dt)
        """
        density_error = 0.0
        for i in range(self.fluid.num_particles):
            density_error += self.rho_derivative[i] * self.fluid.rest_density

        return density_error / self.fluid.num_particles

    @ti.kernel
    def compute_kappa(self):
        """
        compute kappa = alpha * (rho^* - rho_0) / dt
        """
        delta_t_inv = 1 / self.fluid.time_step
        for i in range(self.fluid.num_particles):
            self.kappa[i] = (self.rho_star[i] - 1.0) * self.alpha[i] * delta_t_inv 
            # print("kappa", self.kappa[i])
    
    def correct_density_error(self):
        """
        correct density error
        """
        self.compute_rho_star()
        iteration = 0
        average_density_error = 0.0
        
        while iteration < 1 or iteration < self.fluid.m_max_iterations:
            self.compute_kappa()
            self.correct_density_error_step()
            self.compute_rho_star()
            average_density_error = self.compute_density_error()
            eta = self.fluid.max_error
            # print(average_density_error, eta)
            iteration += 1
            if average_density_error <= eta:
                break
            
        print(f"DFSPH - iteration: {iteration} Avg density err: {average_density_error}")
    
    
    @ti.kernel
    def correct_density_error_step(self):
        for i in range(self.fluid.num_particles):
            k_i = self.kappa[i]
            ret = ti.Vector([0.0, 0.0, 0.0])
            pos_i = self.fluid.positions[i]
            den_i = self.fluid.densities[i]
            for j in range(self.neighbour_num[i]):
                p_j = self.neighbour[i, j]
                if self.is_fluid[p_j]:
                    k_j = self.kappa[p_j]
                    k_sum = k_i + k_j
                    
                    if ti.abs(k_sum) > self.fluid.m_eps * self.fluid.time_step:
                        den_j = self.fluid.densities[p_j]
                        r = self.fluid.positions[i] - self.fluid.positions[p_j]
                        grad_p_j = self.fluid.mass[p_j] * self.fluid.kernel_grad(r)
                        ret -= grad_p_j * (k_i / den_i + k_j / den_j)
                else:
                    k_j = k_i
                    
                    if ti.abs(k_j) > self.fluid.m_eps * self.fluid.time_step:            
                        pos_j = self.rigid_positions[p_j - self.fluid.num_particles]
                        r = pos_i - pos_j
                        grad_p_j = self.rigid_masses[p_j - self.fluid.num_particles] * self.fluid.kernel_grad(r)
                        ret -= grad_p_j * (k_i / den_i)
                        force_j = grad_p_j * (k_i / den_i) * self.fluid.mass[i] / self.fluid.time_step
                        self.rigid.apply_internal_force(force_j, pos_j)
            # print(ret)
            self.fluid.velocities[i] += ret
    
    @ti.kernel
    def compute_density_error(self) -> float:
        """
        compute average of (rho^* / rho_0) - 1
        """
        density_error = 0.0
        for i in range(self.fluid.num_particles):
            density_error += self.rho_star[i] - 1
        return density_error / self.fluid.num_particles
    
    @ti.func
    def compute_forces_rigid(self, i, j):
        m_ij = self.rigid_masses[j - self.fluid.num_particles]
        v_xy = self.fluid.velocities[i] - self.rigid_velocities[j - self.fluid.num_particles]
        R = self.fluid.positions[i] - self.rigid_positions[j - self.fluid.num_particles]
        v_xy = ti.math.dot(v_xy, R)
        nabla_ij = self.fluid.kernel_grad(R)
        
        viscosity_force = 2 * 5 * self.fluid.viscosity * m_ij / self.fluid.densities[i] / (R.norm() ** 2 + 0.01 * self.fluid.h ** 2) * v_xy * nabla_ij / self.fluid.rest_density
        
        force = viscosity_force
        self.fluid.forces[i] += force
        
        force_j = - force * self.fluid.mass[i] / self.rigid_masses[j - self.fluid.num_particles]
        pos_j = self.rigid_positions[j - self.fluid.num_particles]
        self.rigid.apply_internal_force(force_j, pos_j)
        
    @ti.kernel
    def compute_non_pressure_forces(self):
        for i in range(self.fluid.num_particles):
            self.fluid.forces[i] = ti.Vector([0.0, 0.0, 0.0])
            for j in range(self.neighbour_num[i]):
                p_j = self.neighbour[i, j]
                if self.is_fluid[p_j]:
                    self.fluid.compute_non_pressure_forces(i, p_j)
                else:
                    self.compute_forces_rigid(i, p_j)
            self.fluid.forces[i] += self.fluid.gravity[None]
            self.fluid.forces[i] *= self.fluid.mass[i]
        
    
    def update(self):
        self.compute_non_pressure_forces()
        self.fluid.update_velocity()
        self.correct_density_error()
        
        self.fluid.update_position()
        
        self.enforce_domain_boundary()
        
        self.prepare()
        self.correct_divergence()
    
    def prepare(self):
        self.empty_grid()
        self.update_grid()
        self.update_neighbour()
        self.update_rigid_particle_volume()
        
        self.compute_density()
        self.compute_alpha()
    
    def step(self):
        if self.rigid is not None:
            self.get_rigid_pos()
        self.update()