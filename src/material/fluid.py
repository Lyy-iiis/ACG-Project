import taichi as ti
import numpy as np
from .utils import *

@ti.data_oriented
class Fluid:
    def __init__(self, domain_size, mesh, 
                 position=np.array([0.0, 0.0, 0.0]),
                 gravity=np.array([0.0, -9.8, 0.0]), 
                 viscosity=0.1, rest_density=1000.0,
                 time_step=0.00001):
        # self.num_particles = num_particles
        self.domain_size = domain_size
        self.gravity = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.gravity[None] = gravity
        self.viscosity = viscosity
        self.rest_density = rest_density
        self.time_step = time_step

        self.mesh = mesh
        self.volume = ti.field(dtype=ti.f32, shape=())
        self.particle_radius = 0.01
        self.B = 1117.0
        self.gamma = 7.0
        self.stiffness = 50000.0
        self.h = 0.04
        self.mesh.vertices += position
        
        self.volume[None] = self.compute_volume()
        self.original_positions = ti.Vector(position)

        self.grid_size = 0.02
        self.grid_dict = {} # grid_num -> particle_num
        
        self.init_pos()
        self.init_mass()
        self.init_velocity_and_density()
        print("Fluid initialized successfully")
    
    def init_pos(self):
        useful_grid = []
        grid_num = 0
        num_particles = 0
        min_x, min_y, min_z = np.min(self.mesh.vertices, axis=0)
        max_x, max_y, max_z = np.max(self.mesh.vertices, axis=0)
        self.grid_x = int((max_x - min_x) / self.grid_size) + 1
        self.grid_y = int((max_y - min_y) / self.grid_size) + 1
        self.grid_z = int((max_z - min_z) / self.grid_size) + 1
        
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
            
    @ti.kernel
    def init_random_particles(self):
        for i in range(self.num_particles):
            self.positions[i] = ti.Vector([
                ti.random() * self.domain_size[0],
                ti.random() * self.domain_size[1],
                ti.random() * self.domain_size[2]
            ])
        
    @ti.kernel
    def init_velocity_and_density(self):
        for i in range(self.num_particles):
            self.velocities[i] = ti.Vector([0.0, 0.0, 0.0])
            self.densities[i] = self.rest_density
            continue
            for j in range(self.num_particles):
                if i == j:
                    continue
                r = self.positions[i] - self.positions[j]
                r_len = r.norm()
                self.densities[i] += self.kernel_func(r_len) * self.mass[j]
            self.pressures[i] = self.B * ((self.densities[i] / self.rest_density) ** 7 - 1)
            self.forces[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def compute_densities_and_pressures(self):
        for i in range(self.num_particles):
            self.densities[i] = self.rest_density
            self.pressures[i] = 0.0
            continue
            for j in range(self.num_particles):
                if i == j:
                    continue
                r = self.positions[i] - self.positions[j]
                r_len = r.norm()
                # if self.kernel_func(r_len) != 0:
                    # print(self.kernel_func(r_len))
                self.densities[i] += self.kernel_func(r_len) * self.mass[j]
            self.pressures[i] = self.B * ((self.densities[i] / self.rest_density) ** 1 - 1)
            print(self.densities[i])

    @ti.kernel
    def compute_forces(self):
        for i in range(self.num_particles):
            pressure_force = ti.Vector([0.0, 0.0, 0.0])
            viscosity_force = ti.Vector([0.0, 0.0, 0.0])
            for j in range(self.num_particles):
                continue
                if i != j:
                    r = self.positions[i] - self.positions[j]
                    # r_len = r.norm()
                    pressure_force += -self.mass[j] * (self.pressures[i] / self.densities[i] ** 2 + self.pressures[j] / self.densities[j] ** 2) * self.kernel_grad(r)
                    viscosity_force += self.viscosity * self.mass[j] * (self.velocities[j] - self.velocities[i]) / self.densities[j] * self.kernel_grad(r).norm()
            self.forces[i] = pressure_force + viscosity_force + self.mass[i] * self.gravity[None]
            # print(pressure_force, viscosity_force, self.mass[i] * self.gravity[None])

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