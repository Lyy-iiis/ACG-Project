from src.material.fluid import *

@ti.data_oriented
class Container:
    def __init__(self, width, height, depth, fluid: Fluid):
        self.width = width
        self.height = height
        self.depth = depth
        self.fluid = fluid
        self.offset = fluid.original_positions
        self.h = fluid.h

    @ti.func
    def is_within_bounds(self, position: ti.types.vector(3, ti.f32)) -> ti.i32:
        x, y, z = position
        return 0 <= x <= self.width and 0 <= y <= self.height and 0 <= z <= self.depth

    @ti.kernel
    def update(self):
        time_step = self.fluid.time_step
        for i in range(self.fluid.num_particles):
            future_pos = self.fluid.positions[i] + self.fluid.velocities[i] * time_step
            x, y, z = future_pos - self.offset
            # print(x, y, z)
            if x > self.width or x < -self.width:
                self.fluid.velocities[i][0] = -self.fluid.velocities[i][0]
            if y > self.height or y < -self.height:
                self.fluid.velocities[i][1] = -self.fluid.velocities[i][1]
            if z > self.depth or z < -self.depth:
                self.fluid.velocities[i][2] = -self.fluid.velocities[i][2]