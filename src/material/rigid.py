import taichi as ti
import trimesh

def get_rigid_from_mesh(filename):
    mesh = trimesh.load_mesh(filename)
    return mesh

@ti.data_oriented
class RigidBody:
    def __init__(self, mesh, mass=1.0):
        self.mesh = mesh
        self.mass = mass
        self.position = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.torque = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.orientation = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        self.angular_velocity = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.angular_momentum = ti.Vector.field(3, dtype=ti.f32, shape=())

    @ti.kernel
    def apply_force(self, force: ti.types.vector(3, ti.f32), point: ti.types.vector(3, ti.f32)):
        self.force[None] += force
        self.torque[None] += point.cross(force)

    @ti.kernel
    def update(self, dt: ti.f32):
        # Linear motion
        acceleration = self.force[None] / self.mass
        self.velocity[None] += acceleration * dt
        self.position[None] += self.velocity[None] * dt

        # Angular motion
        angular_acceleration = self.torque[None] / self.mass  # Simplified, should use inertia tensor
        self.angular_velocity[None] += angular_acceleration * dt
        self.orientation[None] += self.angular_velocity[None] * dt  # Simplified, should use quaternion or matrix exponential

        # Reset forces and torques
        self.force[None] = ti.Vector([0.0, 0.0, 0.0])
        self.torque[None] = ti.Vector([0.0, 0.0, 0.0])