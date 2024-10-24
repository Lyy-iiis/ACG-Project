import taichi as ti
import numpy as np
from .utils import *

@ti.data_oriented
class Fluid:
    def __init__(self, mesh, 
                 position=np.array([0.0, 0.0, 0.0]),
                 gravity=np.array([0.0, -9.8, 0.0]),
                 viscosity=0.1, rest_density=1000.0,
                 time_step=2e-3):
        # self.num_particles = num_particles
        self.gravity = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.gravity[None] = gravity
        self.viscosity = viscosity
        self.rest_density = rest_density
        self.time_step = time_step

        self.mesh = mesh
        self.volume = ti.field(dtype=ti.f32, shape=())
        self.particle_radius = 0.01
        self.gamma = 7.0
        self.stiffness = 50000.0
        self.h = 0.04
        self.mesh.vertices += position
        
        self.volume[None] = self.compute_volume()
        self.original_positions = ti.Vector(position)

        self.grid_dict = {} # grid_num -> particle_num
        
        self.init_pos()
        self.init_mass()
        self.init_velocity_and_density()
        print("Fluid initialized successfully")
    
    def init_pos(self):
        useful_grid = []
        self.grid_size = 0.01
        grid_num = 0
        num_particles = 0
        min_x, min_y, min_z = np.min(self.mesh.vertices, axis=0)
        max_x, max_y, max_z = np.max(self.mesh.vertices, axis=0)
        self.grid_x = int((max_x - min_x) / self.grid_size) + 1 # number of grids in x direction
        self.grid_y = int((max_y - min_y) / self.grid_size) + 1 # number of grids in y direction
        self.grid_z = int((max_z - min_z) / self.grid_size) + 1 # number of grids in z direction
        
        self.grid_size_x = (max_x - min_x) / (self.grid_x - 1)
        self.grid_size_y = (max_y - min_y) / (self.grid_y - 1)
        self.grid_size_z = (max_z - min_z) / (self.grid_z - 1)

        for x in np.linspace(min_x, max_x, self.grid_x):
            for y in np.linspace(min_y, max_y, self.grid_y):
                for z in np.linspace(min_z, max_z, self.grid_z):
                    if self.mesh.contains(np.array([[x, y, z]])):
                        useful_grid.append(np.array([x, y, z]))
                        self.grid_dict[grid_num] = num_particles
                        num_particles += 1
                    grid_num += 1

        self.num_particles = num_particles
        self.mass = ti.field(dtype=ti.f32, shape=num_particles)
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
        self.densities = ti.field(dtype=ti.f32, shape=num_particles)
        self.pressures = ti.field(dtype=ti.f32, shape=num_particles)
        self.forces = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
        
        num_particles = 0
        for pos in useful_grid:
            self.positions[num_particles] = pos
            num_particles += 1
        print(f"Number of particles: {num_particles}")
        
    def init_mass(self):
        for i in range(self.num_particles):
            self.mass[i] = self.rest_density * self.volume[None] / self.num_particles
    
    def compute_volume(self):
        mesh_volume = 0.0
        
        for i in range(self.mesh.faces.shape[0]):
            a = self.mesh.vertices[self.mesh.faces[i][0]]
            b = self.mesh.vertices[self.mesh.faces[i][1]]
            c = self.mesh.vertices[self.mesh.faces[i][2]]
            volume = np.dot(a, np.cross(b, c)) / 6
            mesh_volume += volume
        print("Mesh volume: ", mesh_volume)
        return mesh_volume
    
    @ti.func
    def kernel_func(self, R_mod):
        # cubic kernel
        res = ti.cast(0.0, ti.f32)
        h = self.h # kernel radius
        k = 8 / np.pi
        k /= h ** 3
        q = R_mod / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res

    @ti.func
    def kernel_grad(self, R):
        # cubic kernel gradient
        res = ti.Vector([0.0 for _ in range(3)])
        h = self.h
        k = 8 / np.pi
        k = 6. * k / h ** 3
        R_mod = R.norm()
        q = R_mod / h
        if R_mod > 1e-5 and q <= 1.0:
            grad_q = R / (R_mod * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res
    
    # @ti.func
    # def neighbour(self, pos):
    #     x, y, z = pos
    #     delta = int(self.h / self.grid_size) + 1
    #     x = int((x - self.min_x) / self.grid_size)
    #     y = int((y - self.min_y) / self.grid_size)
    #     z = int((z - self.min_z) / self.grid_size)
    #     for i in range(-delta, delta):
    #         for j in range(-delta, delta):
    #             for k in range(-delta, delta):
    #                 if 0 <= x + i < self.grid_x and 0 <= y + j < self.grid_y and 0 <= z + k < self.grid_z:
    #                     grid_num = self.pos_to_id(x + i, y + j, z + k)
    #                     if grid_num in self.grid_dict:
    #                         yield self.grid_dict[grid_num]
       
    # @ti.func 
    # def pos_to_id(self, x, y, z):
    #     x_id = (x - self.min_x) / self.grid_size_x
    #     y_id = (y - self.min_y) / self.grid_size_y
    #     z_id = (z - self.min_z) / self.grid_size_z
    #     grid_id = x_id + y_id * self.grid_x + z_id * self.grid_x * self.grid_y
    #     return grid_id
                
        
    @ti.kernel
    def init_velocity_and_density(self):
        for i in range(self.num_particles):
            self.velocities[i] = ti.Vector([0.0, 0.0, 0.0])
            self.densities[i] = self.rest_density

    @ti.kernel
    def compute_densities_and_pressures(self):
        total_density = 0.0
        for i in range(self.num_particles):
            self.densities[i] = 0.0
            self.pressures[i] = 0.0
            for j in range(self.num_particles):
                r = self.positions[i] - self.positions[j]
                r_len = r.norm()
                self.densities[i] += self.kernel_func(r_len) * self.mass[j]
            self.densities[i] = ti.max(self.densities[i], self.rest_density)
            self.pressures[i] = self.stiffness * ((self.densities[i] / self.rest_density) ** 7 - 1)
            total_density += self.densities[i]
        print("average: ", total_density / self.num_particles)

    @ti.kernel
    def compute_forces(self):
        for i in range(self.num_particles):
            pressure_force = ti.Vector([0.0, 0.0, 0.0])
            viscosity_force = ti.Vector([0.0, 0.0, 0.0])
            for j in range(self.num_particles):
                r = self.positions[i] - self.positions[j]
                # r_len = r.norm()
                nabla_ij = self.kernel_grad(r)
                
                pressure_force += -self.mass[j] * (self.pressures[i] / self.densities[i] ** 2 + self.pressures[j] / self.densities[j] ** 2) * nabla_ij
                v_xy = ti.math.dot(self.velocities[i] - self.velocities[j], r)
                m_ij = (self.mass[i] + self.mass[j]) / 2
                viscosity_force += 2 * self.viscosity * m_ij / self.densities[j] / (r.norm() ** 2 + 0.01 * self.h ** 2) * v_xy * nabla_ij
                
            self.forces[i] = pressure_force + viscosity_force + self.gravity[None]
            self.forces[i] *= self.mass[i]

    @ti.kernel
    def update_particles(self):
        x = ti.Vector([0.0, 0.0, 0.0])
        for i in range(self.num_particles):
            self.velocities[i] += self.forces[i] * self.time_step / self.mass[i]
            self.positions[i] += self.velocities[i] * self.time_step
            x += self.positions[i]
        print("average position: ", x / self.num_particles)

    def step(self):
        self.compute_densities_and_pressures()
        self.compute_forces()
        self.update_particles()
        
    def positions_to_ply(self, output_path):
        positions = self.positions.to_numpy()
        write_ply(positions, output_path)