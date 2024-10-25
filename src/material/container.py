from src.material.fluid import *

@ti.data_oriented
class Container:
    def __init__(self, width, height, depth, fluid: Fluid):
        self.domain_size = np.array([width, height, depth])
        self.fluid = fluid
        self.offset = fluid.original_positions
        self.h = fluid.h
        
        # self.grid = ti.field(dtype=ti.i32)
        # ti.root.dense(ti.ijk, (width, height, depth)).dynamic(ti.l, 1024).place(self.grid)

    @ti.kernel
    def enforce_domain_boundary(self):
        # make sure the particles are inside the domain
        for p_i in range(self.fluid.num_particles):
            collision_normal = ti.Vector([0.0, 0.0, 0.0])
            pos = self.fluid.positions[p_i]
            for i in ti.static(range(3)):
                if pos[i] < self.h + self.offset[i] - self.domain_size[i]:
                    self.fluid.positions[p_i][i] = self.h + self.offset[i] - self.domain_size[i]
                    collision_normal[i] += 1.0
                if pos[i] > self.domain_size[i] - self.h + self.offset[i]:
                    self.fluid.positions[p_i][i] = self.domain_size[i] - self.h + self.offset[i]
                    collision_normal[i] += -1.0

            collision_normal_length = collision_normal.norm()
            if collision_normal_length > 1e-6:
                self.simulate_collisions(
                        p_i, collision_normal / collision_normal_length)
    
    @ti.func
    def simulate_collisions(self, p_i, vec):
        c_f = 0.5
        self.fluid.velocities[p_i] -= (1.0 + c_f) * ti.math.dot(self.fluid.velocities[p_i],vec) * vec

    # @ti.func
    # def update_grid(self):
    #     pass
    #     for p_i in range(self.fluid.num_particles):
    #         grid_pos = ti.cast(self.fluid.positions[p_i] / self.h, ti.i32)
    #         self.grid[grid_pos] = p_i