import taichi as ti
import numpy as np
from src.material.geometry import *

@ti.data_oriented
class Cloth:
    def __init__(self, num_particles_x=100, num_particles_y=100,
                 particle_mass=0.1, 
                 structural_stiffness=1e5,
                 shear_stiffness=3e4, 
                 bend_stiffness=3e4,
                 damping=0,
                 gravity=np.array([0.0, -9.8, 0.0]),
                 cloth_size=(0.4, 0.4),
                 initial_position=np.array([0.0, 0.0, 0.0]),
                 restitution_coefficient=0.0,
                 fix=0,
                 collision_radius=0.001,
                 sphere_center=np.array([0.0, 0.4, -2.0]),
                 sphere_radius=0.1,
                 sphere_mass=10.0,
                 sphere_initial_velocity=np.array([0.0, 0.0, 0.0]),
                 time_step=3e-5, fps=60):
        self.num_particles_x = num_particles_x
        self.num_particles_y = num_particles_y
        self.num_particles = num_particles_x * num_particles_y
        self.particle_mass = particle_mass
        self.structural_stiffness = structural_stiffness
        self.shear_stiffness = shear_stiffness
        self.bend_stiffness = bend_stiffness
        self.damping = damping
        self.cloth_size = cloth_size
        self.initial_position = initial_position
        self.restitution_coefficient = restitution_coefficient
        self.gravity = gravity
        self.time_step = time_step
        self.fps = fps
        
        # Basic Cloth and Rigid Coupling: Cloth and Sphere
        # self.sphere_center = ti.Vector.field(3, dtype=ti.f32, shape=())
        # self.sphere_center[None] = ti.Vector(sphere_center)
        # self.sphere_radius = ti.field(dtype=ti.f32, shape=())
        # self.sphere_radius[None] = sphere_radius
        self.sphere_mass = ti.field(dtype=ti.f32, shape=())
        self.sphere_center = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.sphere_velocity = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.sphere_radius = ti.field(dtype=ti.f32, shape=())
        self.sphere_angular_velocity = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.sphere_angular_acceleration = ti.Vector.field(3, dtype=ti.f32, shape=())
        
        self.sphere_mass[None] = sphere_mass
        self.sphere_center[None] = ti.Vector(sphere_center)
        self.sphere_velocity[None] = ti.Vector(sphere_initial_velocity)
        self.sphere_radius[None] = sphere_radius
        self.sphere_angular_velocity[None] = ti.Vector([0.0, 0.0, 0.0])
        self.sphere_angular_acceleration[None] = ti.Vector([0.0, 0.0, 0.0])
        self.total_impulse = ti.Vector.field(3, dtype=ti.f32, shape=())
        

        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=(num_particles_x, num_particles_y))
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=(num_particles_x, num_particles_y))
        self.forces = ti.Vector.field(3, dtype=ti.f32, shape=(num_particles_x, num_particles_y))
        
        self.is_fixed = ti.field(dtype=ti.i32, shape=(num_particles_x, num_particles_y))
        self.initialize_fixed_particles(fix)

        num_faces = 2 * (self.num_particles_x - 1) * (self.num_particles_y - 1)
        self.faces = ti.Vector.field(3, dtype=ti.i32, shape=num_faces)

        # Initializes the deflection of the spring connection
        self.structural_spring_offsets = []
        self.shear_spring_offsets = []
        self.bend_spring_offsets = []

        # Initializes the initial length of the spring
        self.structural_rest_length = self.cloth_size[0] / (self.num_particles_x - 1)
        self.shear_rest_length = self.structural_rest_length * ti.sqrt(2)
        self.bend_rest_length = self.structural_rest_length * 2

        self.initialize_spring_offsets()
        # self.initialize_particles_flat()
        self.initialize_particles_wrinkled()
        # self.initialize_particles_wrinkled_z()  ## 
        self.generate_faces()
        
        # Hash grid parameters
        self.collision_radius = collision_radius
        self.grid_cell_size = collision_radius * 6
        self.grid_size = 64
        self.max_particles_per_cell = 64
        self.num_grid_cells = self.grid_size ** 3

        self.grid = ti.field(dtype=ti.i32, shape=(self.num_grid_cells, self.max_particles_per_cell))
        self.grid_count = ti.field(dtype=ti.i32, shape=self.num_grid_cells)
        self.reset_grid()
        

    def initialize_spring_offsets(self):
        # Connection diagram
        #
        #      O
        #      |
        #  O - O - O
        #      |
        #      O
        self.structural_spring_offsets = [
            ti.Vector([1, 0]),
            ti.Vector([-1, 0]),
            ti.Vector([0, 1]),
            ti.Vector([0, -1])
        ]

        # Connection diagram
        #
        #  O   O
        #   \ /
        #    O
        #   / \
        #  O   O
        self.shear_spring_offsets = [
            ti.Vector([1, 1]),
            ti.Vector([-1, -1]),
            ti.Vector([1, -1]),
            ti.Vector([-1, 1])
        ]

        # Connection diagram
        #
        #          O
        #          |
        #          x
        #          |
        #  O - x - O - x - O
        #          |
        #          x
        #          |
        #          O
        self.bend_spring_offsets = [
            ti.Vector([2, 0]),
            ti.Vector([-2, 0]),
            ti.Vector([0, 2]),
            ti.Vector([0, -2])
        ]

    @ti.kernel
    def initialize_particles_flat(self):
        for i, j in self.positions:
            x = self.initial_position[0] + self.cloth_size[0] * i / (self.num_particles_x - 1)
            y = self.initial_position[1]
            z = self.initial_position[2] + self.cloth_size[1] * j / (self.num_particles_y - 1)
            self.positions[i, j] = ti.Vector([x, y, z])
            self.velocities[i, j] = ti.Vector([0.0, 0.0, 0.0])
            self.forces[i, j] = ti.Vector([0.0, 0.0, 0.0])
    
    @ti.kernel
    def initialize_particles_wrinkled(self):
        for i, j in self.positions:
            x = self.initial_position[0] + self.cloth_size[0] * i / (self.num_particles_x - 1)
            y = self.initial_position[1]
            z = self.initial_position[2] + self.cloth_size[1] * j / (self.num_particles_y - 1)
            
            # Add random offsets to create an undulating effect
            random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.00004
            x += random_offset[0]
            z += random_offset[1]
            
            self.positions[i, j] = ti.Vector([x, y, z])
            self.velocities[i, j] = ti.Vector([0.0, 0.0, 0.0])
            self.forces[i, j] = ti.Vector([0.0, 0.0, 0.0])
            
    @ti.kernel
    def initialize_particles_wrinkled_z(self):
        for i, j in self.positions:
            x = self.initial_position[0] + self.cloth_size[0] * i / (self.num_particles_x - 1)
            y = self.initial_position[1] + self.cloth_size[1] * j / (self.num_particles_y - 1)
            z = self.initial_position[2] 
            
            # Add random offsets to create an undulating effect
            random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.00004
            x += random_offset[0]
            y += random_offset[1]
            
            self.positions[i, j] = ti.Vector([x, y, z])
            self.velocities[i, j] = ti.Vector([0.0, 0.0, 0.0])
            self.forces[i, j] = ti.Vector([0.0, 0.0, 0.0])

            
    @ti.kernel
    def initialize_fixed_particles(self, fix: ti.i32):
        for i, j in self.positions:
            self.is_fixed[i, j] = 0  # 0: not fixed, 1: fixed
        if fix == 1:
            # # fix the top left particle
            # self.is_fixed[0, 0] = 1
            # # fix the top right particle
            # self.is_fixed[self.num_particles_x - 1, 0] = 1
            # # fix the bottom left particle
            # self.is_fixed[0, self.num_particles_y - 1] = 1
            # # fix the bottom right particle
            # self.is_fixed[self.num_particles_x - 1, self.num_particles_y - 1] = 1
            pass

    @ti.kernel
    def compute_forces(self):
        for i, j in self.positions:
            # Reset force
            self.forces[i, j] = self.particle_mass * ti.Vector(self.gravity)
            self.forces[i, j] += -self.damping * self.velocities[i, j]

            pos_i = self.positions[i, j]
            vel_i = self.velocities[i, j]

            # structural_spring_offsets
            for offset in ti.static(self.structural_spring_offsets):
                ni = i + offset[0]
                nj = j + offset[1]
                if 0 <= ni < self.num_particles_x and 0 <= nj < self.num_particles_y:
                    pos_j = self.positions[ni, nj]
                    vel_j = self.velocities[ni, nj]
                    self.apply_spring_force(i, j, ni, nj, pos_i, pos_j, vel_i, vel_j, self.structural_stiffness, self.structural_rest_length)

            # shear_spring_offsets
            for offset in ti.static(self.shear_spring_offsets):
                ni = i + offset[0]
                nj = j + offset[1]
                if 0 <= ni < self.num_particles_x and 0 <= nj < self.num_particles_y:
                    pos_j = self.positions[ni, nj]
                    vel_j = self.velocities[ni, nj]
                    self.apply_spring_force(i, j, ni, nj, pos_i, pos_j, vel_i, vel_j, self.shear_stiffness, self.shear_rest_length)

            # bend_spring_offsets
            for offset in ti.static(self.bend_spring_offsets):
                ni = i + offset[0]
                nj = j + offset[1]
                if 0 <= ni < self.num_particles_x and 0 <= nj < self.num_particles_y:
                    pos_j = self.positions[ni, nj]
                    vel_j = self.velocities[ni, nj]
                    self.apply_spring_force(i, j, ni, nj, pos_i, pos_j, vel_i, vel_j, self.bend_stiffness, self.bend_rest_length)

    @ti.func
    def apply_spring_force(self, i, j, ni, nj, pos_i, pos_j, vel_i, vel_j, stiffness, rest_length):
        delta_pos = pos_i - pos_j
        current_length = delta_pos.norm(1e-16)
        direction = delta_pos.normalized(1e-16)
        spring_force = -stiffness * (current_length - rest_length) * direction

        relative_vel = vel_i - vel_j
        damping_force = -self.damping * relative_vel.dot(direction) * direction

        total_force = spring_force + damping_force

        self.forces[i, j] += total_force

    # @ti.kernel
    # def collision_detection(self, rigid_body, dt_old:float):
    #     dt = ti.cast(dt_old, ti.f32)
    #     for i, j in self.positions:
    #         pos = self.positions[i, j]
    #         vel = self.velocities[i, j]
    #         collision, normal = rigid_body.check_collision(pos)
    #         if collision:
    #             # Calculate the relative velocity with respect to a rigid body
    #             rel_vel = vel - rigid_body.get_velocity_at_point(pos)
    #             vel_normal = rel_vel.dot(normal) * normal
    #             vel_tangent = rel_vel - vel_normal
    #             # Update particle velocity
    #             new_rel_vel = vel_tangent - self.restitution_coefficient * vel_normal
    #             self.velocities[i, j] = new_rel_vel + rigid_body.get_velocity_at_point(pos)
    #             # Calculate and apply collision forces
    #             collision_impulse = -(1 + self.restitution_coefficient) * vel_normal * self.particle_mass
    #             collision_force = collision_impulse / dt
    #             rigid_body.apply_internal_force(collision_force, pos)
    
    @ti.kernel
    def collision_with_fixed_sphere(self):
        for i, j in self.positions:
            pos = self.positions[i, j]
            to_particle = pos - self.sphere_center[None]
            dist = to_particle.norm()
            if dist < self.sphere_radius[None]:
                penetration_depth = self.sphere_radius[None] - dist
                
                direction = ti.Vector([0.0, 1.0, 0.0])  # Default value: avoid division by zero
                if dist > 1e-6:
                    direction = to_particle / dist

                # Correction position
                self.positions[i, j] += direction * penetration_depth
                
                # Reflection speed
                vel = self.velocities[i, j]
                vel_normal = vel.dot(direction) * direction
                vel_tangent = vel - vel_normal
                vel_normal *= -self.restitution_coefficient
                self.velocities[i, j] = vel_tangent + vel_normal
                
    @ti.kernel
    def collision_with_sphere(self):
        for i, j in self.positions:
            pos = self.positions[i, j]
            to_particle = pos - self.sphere_center[None]
            dist = to_particle.norm()
            if dist < self.sphere_radius[None]:
                direction = to_particle.normalized()
                penetration_depth = self.sphere_radius[None] - dist
                # Correction position to avoid penetration
                self.positions[i, j] += direction * penetration_depth

                # Calculate relative velocity
                relative_velocity = self.velocities[i, j] - self.sphere_velocity[None]
                normal_velocity = relative_velocity.dot(direction)

                if normal_velocity < 0:
                    # Calculate impulse
                    impulse = -normal_velocity * direction * self.particle_mass
                    # update particle velocity
                    self.velocities[i, j] = self.sphere_velocity[None]
                    # Accumulate impulse
                    ti.atomic_add(self.total_impulse[None], -impulse)
    
        
    @ti.func
    def hash_grid_func(self, x, y, z):
        xi = int(ti.floor(x/ self.grid_cell_size))
        yi = int(ti.floor(y / self.grid_cell_size))
        zi = int(ti.floor(z/ self.grid_cell_size))  ## The center of this scene is at (-0.25, 0.0, -2.25)
        xi = xi % self.grid_size
        yi = yi % self.grid_size
        zi = zi % self.grid_size
        return xi * self.grid_size * self.grid_size + yi * self.grid_size + zi
    
    @ti.kernel
    def reset_grid(self):
        for i in range(self.num_grid_cells):
            self.grid_count[i] = 0

    @ti.kernel
    def update_grid(self):
        for i, j in self.positions:
            pos = self.positions[i, j]
            cell = self.hash_grid_func(pos.x, pos.y, pos.z)
            index = ti.atomic_add(self.grid_count[cell], 1)
            if index < self.max_particles_per_cell:
                self.grid[cell, index] = i * self.num_particles_y + j

    @ti.kernel
    def self_collision(self):
        for idx in range(self.num_particles):
            i = idx // self.num_particles_y
            j = idx % self.num_particles_y

            pos_i = self.positions[i, j]
            if self.is_fixed[i, j] == 1:
                continue  # skip fixed particles

            cell = self.hash_grid_func(pos_i.x, pos_i.y, pos_i.z)
            # print(self.grid_count[cell])
            for offset in range(self.grid_count[cell]):
                idx_j = self.grid[cell, offset]
                if idx_j != idx:
                    ni = idx_j // self.num_particles_y
                    nj = idx_j % self.num_particles_y
                    pos_j = self.positions[ni, nj]

                    delta = pos_i - pos_j
                    dist = delta.norm(1e-6)
                    if dist < self.collision_radius * 2:
                        penetration_depth = self.collision_radius * 2 - dist
                        dir = delta.normalized(1e-6)
                        correction = 0.5 * penetration_depth * dir
                        self.positions[i, j] += correction
                        self.positions[ni, nj] -= correction

    @ti.kernel
    def update(self):
        for i, j in self.positions:
            if self.is_fixed[i, j] == 0:
                acceleration = self.forces[i, j] / self.particle_mass
                self.velocities[i, j] += acceleration * self.time_step
                self.positions[i, j] += self.velocities[i, j] * self.time_step
            else:
                self.velocities[i, j] = ti.Vector([0.0, 0.0, 0.0])
             
    @ti.kernel   
    def update_sphere(self):
        gravity = ti.Vector(self.gravity)
        self.sphere_velocity[None] += gravity * self.time_step
        self.sphere_center[None] += self.sphere_velocity[None] * self.time_step

    @ti.kernel
    def generate_faces(self):
        face_id = 0
        for i in range(self.num_particles_x - 1):
            for j in range(self.num_particles_y - 1):
                idx0 = i * self.num_particles_y + j
                idx1 = idx0 + 1
                idx2 = (i + 1) * self.num_particles_y + j
                idx3 = idx2 + 1

                self.faces[face_id] = ti.Vector([idx0, idx2, idx1])
                face_id += 1

                self.faces[face_id] = ti.Vector([idx1, idx2, idx3])
                face_id += 1

    @ti.kernel
    def get_flat_positions(self, flat_positions: ti.template()):
        for i, j in ti.ndrange(self.num_particles_x, self.num_particles_y):
            flat_positions[i * self.num_particles_y + j] = self.positions[i, j]

    def substep(self):
        self.compute_forces()
        self.update()
        
        # self.total_impulse[None] = ti.Vector([0.0, 0.0, 0.0])
        
        ## Collision detection with fixed spheres
        self.collision_with_fixed_sphere()
        
        # self.collision_with_sphere()
        
        # Self collision
        self.reset_grid()
        self.update_grid()
        self.self_collision()
        
        # self.sphere_velocity[None] += self.total_impulse[None] / self.sphere_mass[None]
        # self.update_sphere()
