import taichi as ti
import numpy as np

@ti.data_oriented
class RigidBody:
    def __init__(self, mesh, mass=1.0, 
                position=np.array([0.0, 0.0, 0.0]), 
                orientation=np.eye(3), 
                velocity=np.array([0.0, 0.0, 0.0]), 
                angular_velocity=np.array([0.0, 0.0, 0.0])):
        # self.mesh = mesh
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=len(mesh.vertices))
        self.faces = ti.Vector.field(3, dtype=ti.i32, shape=len(mesh.faces))
        for i in range(len(mesh.vertices)):
            self.vertices[i] = mesh.vertices[i]
        for i in range(len(mesh.faces)):
            self.faces[i] = mesh.faces[i]

        self.mass, self.volume = mass, ti.field(ti.f32, shape=())
        self.centralize() # centralize the mesh
        self.inertia_tensor = self.inertia() # inertia tensor relative to the center of mass with respect to the canonical frame

        self.position = ti.Vector.field(3, dtype=ti.f32, shape=()) # position of the center of mass
        self.position[None] = position
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=()) # velocity of the center of mass
        self.velocity[None] = velocity
        self.orientation = ti.Matrix.field(3, 3, dtype=ti.f32, shape=()) # orientation matrix of the body
        self.orientation[None] = orientation
        self.angular_velocity = ti.Vector.field(3, dtype=ti.f32, shape=()) # angular velocity of the body
        self.angular_velocity[None] = angular_velocity
        
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.torque = ti.Vector.field(3, dtype=ti.f32, shape=()) # torque relative to the center of mass
        self.angular_momentum = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.eular_angles = ti.Vector.field(3, dtype=ti.f32, shape=())
        
    @ti.func
    def mass_center(self) -> ti.types.vector(3, ti.f32):
        mesh_volume = ti.float32(0.0)
        temp = ti.Vector([0.0, 0.0, 0.0])
        
        for i in range(self.faces.shape[0]):
            # print(self.faces[i][0])
            center = 0.25 * (self.vertices[self.faces[i][0]] + self.vertices[self.faces[i][1]] + self.vertices[self.faces[i][2]])
            volume = ti.abs(ti.math.dot(self.vertices[self.faces[i][0]], ti.math.cross(self.vertices[self.faces[i][1]], self.vertices[self.faces[i][2]]))) / 6
            mesh_volume += volume
            temp += center * volume
        
        self.volume[None] = mesh_volume
        return temp / mesh_volume
    
    @ti.kernel
    def centralize(self):
        center = self.mass_center()
        for i in range(self.vertices.shape[0]):
            self.vertices[i] -= center
    
    @ti.kernel
    def inertia(self) -> ti.types.matrix(3, 3, ti.f32):
        # vertices = self.mesh.vertices
        # faces = self.mesh.faces
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
    def apply_force(self, force: ti.types.vector(3, ti.f32), point: ti.types.vector(3, ti.f32)):
        self.force[None] += force
        self.torque[None] += ti.math.cross(point - self.position[None], force)

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