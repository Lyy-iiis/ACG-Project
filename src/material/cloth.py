import taichi as ti
import numpy as np
from src.material.geometry import *

@ti.data_oriented
class Cloth:
    def __init__(self, num_particles_x=100, num_particles_y=100,
                 particle_mass=0.1, stiffness=3e4, damping=0,
                 gravity=np.array([0.0, -9.8, 0.0]),
                 cloth_size=(0.4, 0.4),
                 initial_position=np.array([0.0, 0.0, 0.0]),
                 restitution_coefficient=0.0,
                 dt=3e-5):
        self.num_particles_x = num_particles_x
        self.num_particles_y = num_particles_y
        self.num_particles = num_particles_x * num_particles_y
        self.particle_mass = particle_mass
        self.stiffness = stiffness
        self.damping = damping
        self.cloth_size = cloth_size
        self.initial_position = initial_position
        self.restitution_coefficient = restitution_coefficient
        self.gravity = gravity
        self.dt = dt

        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=(num_particles_x, num_particles_y))
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=(num_particles_x, num_particles_y))
        self.forces = ti.Vector.field(3, dtype=ti.f32, shape=(num_particles_x, num_particles_y))

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
        self.initialize_particles_flat()
        self.generate_faces()

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
            random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.001
            x += random_offset[0]
            z += random_offset[1]
            
            self.positions[i, j] = ti.Vector([x, y, z])
            self.velocities[i, j] = ti.Vector([0.0, 0.0, 0.0])
            self.forces[i, j] = ti.Vector([0.0, 0.0, 0.0])

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
                    self.apply_spring_force(i, j, ni, nj, pos_i, pos_j, vel_i, vel_j, self.structural_rest_length)

            # shear_spring_offsets
            for offset in ti.static(self.shear_spring_offsets):
                ni = i + offset[0]
                nj = j + offset[1]
                if 0 <= ni < self.num_particles_x and 0 <= nj < self.num_particles_y:
                    pos_j = self.positions[ni, nj]
                    vel_j = self.velocities[ni, nj]
                    self.apply_spring_force(i, j, ni, nj, pos_i, pos_j, vel_i, vel_j, self.shear_rest_length)

            # bend_spring_offsets
            for offset in ti.static(self.bend_spring_offsets):
                ni = i + offset[0]
                nj = j + offset[1]
                if 0 <= ni < self.num_particles_x and 0 <= nj < self.num_particles_y:
                    pos_j = self.positions[ni, nj]
                    vel_j = self.velocities[ni, nj]
                    self.apply_spring_force(i, j, ni, nj, pos_i, pos_j, vel_i, vel_j, self.bend_rest_length)

    @ti.func
    def apply_spring_force(self, i, j, ni, nj, pos_i, pos_j, vel_i, vel_j, rest_length):
        delta_pos = pos_i - pos_j
        current_length = delta_pos.norm(1e-16)
        direction = delta_pos.normalized(1e-16)
        spring_force = -self.stiffness * (current_length - rest_length) * direction

        relative_vel = vel_i - vel_j
        damping_force = -self.damping * relative_vel.dot(direction) * direction

        total_force = spring_force + damping_force

        self.forces[i, j] += total_force

    @ti.kernel
    def collision_detection(self, rigid_body: ti.template(), dt_old:float):
        dt = ti.cast(dt_old, ti.f32)
        for i, j in self.positions:
            pos = self.positions[i, j]
            vel = self.velocities[i, j]

            collision, normal = rigid_body.check_collision(pos)

            if collision:
                # Calculate the relative velocity with respect to a rigid body
                rel_vel = vel - rigid_body.get_velocity_at_point(pos)
                vel_normal = rel_vel.dot(normal) * normal
                vel_tangent = rel_vel - vel_normal

                # Update particle velocity
                new_rel_vel = vel_tangent - self.restitution_coefficient * vel_normal
                self.velocities[i, j] = new_rel_vel + rigid_body.get_velocity_at_point(pos)

                # Calculate and apply collision forces
                collision_impulse = -(1 + self.restitution_coefficient) * vel_normal * self.particle_mass
                collision_force = collision_impulse / dt

                rigid_body.apply_internal_force(collision_force, pos)

    @ti.kernel
    def update(self):
        # dt = ti.cast(dt_old, ti.f32)
        for i, j in self.positions:
            acceleration = self.forces[i, j] / self.particle_mass
            self.velocities[i, j] += acceleration * self.dt
            self.positions[i, j] += self.velocities[i, j] * self.dt

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
        # self.collision_detection(rigid_body, self.dt)
        self.update()
