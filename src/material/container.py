from src.material.fluid import *
from src.material.rigid import *

@ti.data_oriented
class Container:
    def __init__(self, width, height, depth, fluid: Fluid, rigid: RigidBody):
        self.domain_size = ti.Vector([width, height, depth])
        self.fluid = fluid
        self.rigid = rigid
        self.offset = fluid.original_positions
        self.h = fluid.h
        self.max_num_particles = self.fluid.num_particles
        
        self.grid = ti.field(dtype=ti.i32)
        self.grid_num = ti.field(dtype=ti.i32)
        self.grid_x = int(width / fluid.grid_size_x) + 1
        self.grid_y = int(height / fluid.grid_size_y) + 1
        self.grid_z = int(depth / fluid.grid_size_z) + 1
        self.grid_size_x = 2 * width / (self.grid_x - 1)
        self.grid_size_y = 2 * height / (self.grid_y - 1)
        self.grid_size_z = 2 * depth / (self.grid_z - 1)
        
        self.idx_to_grid = ti.Vector.field(3, dtype=ti.i32, shape=(self.max_num_particles,))
        
        ti.root.dense(ti.ijk, (self.grid_x, self.grid_y, self.grid_z)).dynamic(ti.l, 1024).place(self.grid)
        ti.root.dense(ti.ijk, (self.grid_x, self.grid_y, self.grid_z)).place(self.grid_num)
        
        self.neighbour = ti.field(dtype=ti.i32)
        ti.root.dense(ti.i, self.max_num_particles).dynamic(ti.j, 2048).place(self.neighbour)
        self.neighbour_num = ti.field(dtype=ti.i32, shape=self.max_num_particles)
        # self.update()
        print("Container initialized successfully", self.grid_num.shape)

    @ti.func
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
    
    @ti.func
    def empty_grid(self):
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                for z in range(self.grid_z):
                    self.grid[int(x), int(y), int(z)].deactivate()
                    self.grid_num[x, y, z] = 0

        for i in range(self.fluid.num_particles):
            self.neighbour_num[int(i)] = 0
            self.neighbour[int(i)].deactivate()
            
        # for i in range(self.rigid.num_particles+self.fluid.num_particles):
        #     self.neighbour_num[int(i)] = 0
        #     self.neighbour[i].deactivate()
    
    def get_rigid_pos(self):
        self.rigid_positions_np = self.rigid.get_voxel()[0]
        self.rigid_positions = ti.Vector.field(3, dtype=ti.f32, shape=self.rigid_positions_np.shape[0])
        self.rigid_positions.from_numpy(self.rigid_positions_np)

    @ti.func
    def update_grid(self):        
        for p_i in range(self.fluid.num_particles):
            pos = self.fluid.positions[p_i] - self.offset + self.domain_size
            x_id = int(pos[0] / self.grid_size_x)
            y_id = int(pos[1] / self.grid_size_y)
            z_id = int(pos[2] / self.grid_size_z)
            self.grid[x_id, y_id, z_id].append(p_i)
            self.grid_num[x_id, y_id, z_id] += 1
            self.idx_to_grid[p_i] = ti.Vector([x_id, y_id, z_id])
        
        # for p_i in range(self.rigid.num_particles):
        #     pos = self.rigid_positions[p_i] - self.offset + self.domain_size
        #     x_id = int(pos[0] / self.grid_size_x)
        #     y_id = int(pos[1] / self.grid_size_y)
        #     z_id = int(pos[2] / self.grid_size_z)
        #     self.grid[x_id, y_id, z_id].append(p_i+self.fluid.num_particles)
        #     self.grid_num[x_id, y_id, z_id] += 1
        #     self.idx_to_grid[p_i+self.fluid.num_particles] = ti.Vector([x_id, y_id, z_id])
    
    @ti.func
    def update_neighbour(self):
        for p_i in range(self.fluid.num_particles):
            grid_idx = self.idx_to_grid[p_i]
            
            for offset in ti.grouped(ti.ndrange((-2, 3), (-2, 3), (-2, 3))):
                neighbor_idx = grid_idx + offset
                if 0 <= neighbor_idx[0] < self.grid_x and 0 <= neighbor_idx[1] < self.grid_y and 0 <= neighbor_idx[2] < self.grid_z:
                    for j in range(self.grid_num[neighbor_idx]):
                        p_j = self.grid[neighbor_idx,j]
                        r_len = (self.fluid.positions[p_j] - self.fluid.positions[p_i]).norm()
                        if r_len <= self.fluid.h:
                            self.neighbour[p_i].append(p_j)
                            self.neighbour_num[p_i] += 1
        
        # for p_i in range(self.rigid.num_particles):
        #     grid_idx = self.idx_to_grid[p_i+self.fluid.num_particles]
            
        #     for offset in ti.grouped(ti.ndrange((-2, 3), (-2, 3), (-2, 3))):
        #         neighbor_idx = grid_idx + offset
        #         if 0 <= neighbor_idx[0] < self.grid_x and 0 <= neighbor_idx[1] < self.grid_y and 0 <= neighbor_idx[2] < self.grid_z:
        #             for j in range(self.grid_num[neighbor_idx+self.fluid.num_particles]):
        #                 p_j = self.grid[neighbor_idx,j]
        #                 if p_j < self.fluid.num_particles:
        #                     r_len = (self.fluid.positions[p_j] - self.rigid_positions[p_i]).norm()
        #                 elif p_j < self.fluid.num_particles+self.rigid.num_particles:
        #                     r_len = (self.rigid_positions[p_j-self.fluid.num_particles] - self.rigid_positions[p_i]).norm()
        #                 else:
        #                     assert False
                        
        #                 if r_len <= self.fluid.h:
        #                     self.fluid.neighbour[p_i].append(p_j)
        #                     self.fluid.neighbour_num[p_i] += 1
    
    # @ti.func
    # def compute_for_neighbour(self, p_i, task: ti.template()):
    #     for j in range(self.neighbour_num[p_i]):
    #         p_j = self.neighbour[p_i,j]
    #         task(p_i, p_j)
    
    @ti.func
    def compute_densities_and_pressures(self):
        for i in range(self.fluid.num_particles):
            self.fluid.densities[i] = 0.0
            self.fluid.pressures[i] = 0.0
            for j in range(self.neighbour_num[i]):
                p_j = self.neighbour[i,j]
                self.fluid.compute_densities_and_pressures(i, p_j)            
            self.fluid.densities[i] = ti.max(self.fluid.densities[i], self.fluid.rest_density)
            self.fluid.pressures[i] = self.fluid.stiffness * ((self.fluid.densities[i] / self.fluid.rest_density) ** 7 - 1)

        self.fluid.avg_density[None] = 0.0
        for i in range(self.fluid.num_particles):
            self.fluid.avg_density[None] += self.fluid.densities[i]
        self.fluid.avg_density[None] /= self.fluid.num_particles
        
    @ti.func
    def compute_forces(self):
        for i in range(self.fluid.num_particles):
            self.fluid.forces[i] = ti.Vector([0.0, 0.0, 0.0])
            for j in range(self.neighbour_num[i]):
                p_j = self.neighbour[i,j]
                self.fluid.compute_forces(i, p_j)
            self.fluid.forces[i] += self.fluid.gravity[None]
            self.fluid.forces[i] *= self.fluid.mass[i]
    
    @ti.kernel
    def update(self):
        self.empty_grid()
        self.enforce_domain_boundary()
        # self.get_rigid_pos()
        self.update_grid()
        self.update_neighbour()
        self.compute_densities_and_pressures()
        self.compute_forces()
        self.fluid.update_particles()