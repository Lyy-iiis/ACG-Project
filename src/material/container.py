from src.material.fluid import *

@ti.data_oriented
class Container:
    def __init__(self, width, height, depth, fluid: Fluid):
        self.width = width
        self.height = height
        self.depth = depth
        self.fluid = fluid

    @ti.func
    def is_within_bounds(self, position: ti.types.vector(3, ti.f32)) -> ti.i32:
        x, y, z = position
        return 0 <= x <= self.width and 0 <= y <= self.height and 0 <= z <= self.depth

    @ti.kernel
    def update(self):
        time_step = self.fluid.time_step
        for i in range(self.fluid.num_particles):
            self.handle_boundary_collision(self.fluid.positions[i])
            self.fluid.positions[i] += self.fluid.velocities[i] * time_step

    @ti.kernel
    def handle_boundary_collision(self, position: ti.types.vector(3, ti.f32)):
        if not self.is_within_bounds(position):
            x, y, z = position
            x = min(max(x, 0), self.width)
            y = min(max(y, 0), self.height)
            z = min(max(z, 0), self.depth)
            position[0] = x
            position[1] = y
            position[2] = z