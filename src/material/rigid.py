import taichi as ti
import numpy as np
from src.material.geometry import *
import time

@ti.data_oriented
class RigidBody:
    def __init__(self, type=None, mesh=None, mass=1.0, 
                radius=1.0, height=1.0, center=[0.0, 0.0, 0.0],size=[1.0, 1.0],
                inner_radius=0.5, outer_radius=1.0, resolution=100,
                position=np.array([0.0, 0.0, 0.0]), 
                orientation=np.eye(3), 
                velocity=np.array([0.0, 0.0, 0.0]), 
                angular_velocity=np.array([0.0, 0.0, 0.0]),
                collision_threshold=np.finfo(np.float32).tiny):

        if type == 'Ball':
            mesh = Ball(radius, center, resolution)
        elif type == 'Box':
            mesh = Box(size, center)
        elif type == 'Cylinder':
            mesh = Cylinder(radius, height, center, resolution)
        elif type == 'Plane':
            mesh = Plane(size, center)
        elif type == 'Sphere':
            mesh = Sphere(radius, center, resolution)
        elif type == 'Cone':
            mesh = Cone(radius, height, center, resolution)
        elif type == 'Torus':
            mesh = Torus(inner_radius, outer_radius, center, resolution)
        elif mesh is None:
            raise ValueError("Please provide a mesh or a type of geometry")
        
        self.mesh = mesh
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=len(mesh.vertices))
        self.faces = ti.Vector.field(3, dtype=ti.i32, shape=len(mesh.faces))
        for i in range(len(mesh.vertices)):
            self.vertices[i] = mesh.vertices[i]
        for i in range(len(mesh.faces)):
            self.faces[i] = mesh.faces[i]
        self.mass, self.volume = mass, ti.field(ti.f32, shape=())
        self.mass_center_offset = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.centralize() # centralize the mesh
        self.inertia_tensor = self.inertia() # inertia tensor relative to the center of mass with respect to the canonical frame
        self.mesh = trimesh.Trimesh(vertices=self.vertices.to_numpy(), faces=self.faces.to_numpy())
        self.voxel = None
        self.get_voxel()
        self.position = ti.Vector.field(3, dtype=ti.f32, shape=()) # position of the center of mass
        self.position[None] = position
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=()) # velocity of the center of mass
        self.velocity[None] = velocity
        self.orientation = ti.Matrix.field(3, 3, dtype=ti.f32, shape=()) # orientation matrix of the body
        self.orientation[None] = orientation
        self.angular_velocity = ti.Vector.field(3, dtype=ti.f32, shape=()) # angular velocity of the body
        self.angular_velocity[None] = angular_velocity
        self.collision_threshold = collision_threshold
        self.num_particles = 0
        
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.torque = ti.Vector.field(3, dtype=ti.f32, shape=()) # torque relative to the center of mass
        self.angular_momentum = ti.Vector.field(3, dtype=ti.f32, shape=())
        # self.eular_angles = ti.Vector.field(3, dtype=ti.f32, shape=())
        
    @ti.func
    def mass_center(self) -> ti.types.vector(3, ti.f32):
        mesh_volume = ti.float32(0.0)
        temp = ti.Vector([0.0, 0.0, 0.0])
        
        for i in range(self.faces.shape[0]):
            # print(self.faces[i][0])
            center = 0.25 * (self.vertices[self.faces[i][0]] + self.vertices[self.faces[i][1]] + self.vertices[self.faces[i][2]])
            volume = ti.math.dot(self.vertices[self.faces[i][0]], ti.math.cross(self.vertices[self.faces[i][1]], self.vertices[self.faces[i][2]])) / 6
            mesh_volume += volume
            temp += center * volume
        
        self.volume[None] = ti.abs(mesh_volume)
        return temp / mesh_volume
    
    @ti.kernel
    def centralize(self):
        center = self.mass_center()
        self.mass_center_offset[None] = center
        for i in range(self.vertices.shape[0]):
            self.vertices[i] -= center
    
    def get_voxel(self):
        mesh = self.mesh.copy()
        # mesh.apply_transform(np.vstack((np.hstack((self.orientation.to_numpy(), self.position.to_numpy().reshape(-1, 1))), [0, 0, 0, 1])))
        self.voxel = mesh.voxelized(pitch=0.01).fill().points.astype(np.float32)
        self.num_particles = self.voxel.shape[0]
    
    @ti.kernel
    def inertia(self) -> ti.types.matrix(3, 3, ti.f32):
        covarience_tensor = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        canoical_inertia_tensor = ti.Matrix([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]) / 60
        
        for i in range(self.faces.shape[0]):
            # ti.static_print(self.faces[i, 0])
            transform = ti.Matrix.cols([self.vertices[self.faces[i][0]], self.vertices[self.faces[i][1]], self.vertices[self.faces[i][2]]])
            covarience_tensor += transform.determinant() * transform @ canoical_inertia_tensor @ transform.transpose()
        covarience_tensor *= self.mass / self.volume[None]
        inertia_tensor = ti.Matrix.identity(ti.f32, 3) * ti.Matrix.trace(covarience_tensor) - covarience_tensor
        return inertia_tensor

    @ti.kernel
    def apply_external_force(self, force: ti.types.vector(3, ti.f32), point: ti.types.vector(3, ti.f32)):
        self.force[None] += force
        self.torque[None] += ti.math.cross(point - self.position[None], force)

    @ti.func
    def apply_internal_force(self, force: ti.types.vector(3, ti.f32), point: ti.types.vector(3, ti.f32)):
        self.force[None] += force
        self.torque[None] += ti.math.cross(point - self.position[None], force)

    @ti.func
    def check_collision(self, point: ti.types.vector(3, ti.f32)) -> ti.types.vector(2, ti.f32):
        min_distance = float('inf')  # Used to record the minimum collision distance
        closest_normal = ti.Vector([0.0, 0.0, 0.0])  # Used to record the closest normal

        for i in range(self.faces.shape[0]):
            # Get the three vertices of a triangle
            v0 = self.position[None] + self.orientation[None] @ self.vertices[self.faces[i][0]]
            v1 = self.position[None] + self.orientation[None] @ self.vertices[self.faces[i][1]]
            v2 = self.position[None] + self.orientation[None] @ self.vertices[self.faces[i][2]]

            # Compute the normal vector of the triangle
            normal = (v1 - v0).cross(v2 - v0).normalized()

            # Compute the distance of the point to the face
            distance = abs((point - v0).dot(normal))

            # If the distance is less than the threshold, a collision is considered to have occurred
            if distance < self.collision_threshold:
                if distance < min_distance:
                    min_distance = distance
                    closest_normal = normal

        collision = 0
        normal = ti.Vector([0.0, 0.0, 0.0])
        
        if min_distance < float('inf'):
            collision = 1
            normal = closest_normal

        return collision, normal
        
    @ti.func
    def get_velocity_at_point(self, point: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
        # Velocity coupling relation
        r = point - self.position[None]
        
        linear_velocity = self.velocity[None]
        
        angular_velocity_at_point = self.angular_velocity[None].cross(r)
        
        return linear_velocity + angular_velocity_at_point


    @ti.kernel
    def update(self, dt_old: float):
        # Linear motion
        dt = ti.cast(dt_old, ti.f32)

        acceleration = self.force[None] / self.mass
        self.velocity[None] += acceleration * dt
        self.position[None] += self.velocity[None] * dt

        # Angular motion
        # angular_acceleration = self.torque[None] / self.mass  # Simplified, should use inertia tensor
        angular_acceleration = self.compute_angular_acceleration()
        self.angular_velocity[None] += angular_acceleration * dt
        angular_velocity_norm = self.angular_velocity[None].norm()
        exp_A = ti.Matrix.identity(ti.f32, 3)
        print(angular_velocity_norm)
        if angular_velocity_norm > 1e-8:
            angular_velocity_matrix = ti.Matrix([
                [0, -self.angular_velocity[None][2], self.angular_velocity[None][1]],
                [self.angular_velocity[None][2], 0, -self.angular_velocity[None][0]],
                [-self.angular_velocity[None][1], self.angular_velocity[None][0], 0]
            ]) / angular_velocity_norm

            exp_A = ti.Matrix.identity(ti.f32, 3) + angular_velocity_matrix * ti.sin(angular_velocity_norm * dt) + angular_velocity_matrix @ angular_velocity_matrix * (1 - ti.cos(angular_velocity_norm * dt))
        # print(exp_A)
        self.orientation[None] = exp_A @ self.orientation[None]

        # Reset forces and torques
        self.force[None] = ti.Vector([0.0, 0.0, 0.0])
        self.torque[None] = ti.Vector([0.0, 0.0, 0.0])
        
    @ti.func
    def compute_angular_acceleration(self) -> ti.types.vector(3, ti.f32):
        inertia_tensor_now = self.orientation[None] @ self.inertia_tensor @ self.orientation[None].transpose() # inertia tensor relative to the center of mass with respect to the current frame
        self.angular_momentum[None] = inertia_tensor_now @ self.angular_velocity[None]
        torque = self.torque[None] - ti.math.cross(self.angular_velocity[None], self.angular_momentum[None])
        return inertia_tensor_now.inverse() @ torque
    
    def get_states(self):
        velocity = self.velocity.to_numpy()
        position = self.position.to_numpy()
        orientation = self.orientation.to_numpy()
        angular_velocity = self.angular_velocity.to_numpy()
        return {
            'velocity': velocity,
            'position': position,
            'orientation': orientation,
            'angular_velocity': angular_velocity
        }