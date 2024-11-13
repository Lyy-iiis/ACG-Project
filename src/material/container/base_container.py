from src.material.fluid.basefluid import *
from src.material.rigid import *
import os
from src.material.geometry import *

@ti.data_oriented
class Container:
    def __init__(self, width, height, depth, fluid: BaseFluid, rigid: RigidBody):
        self.domain_size = ti.Vector([width, height, depth])
        self.fluid = fluid
        if rigid is None:
            self.rigid = RigidBody("Ball", radius=0.1, position=np.array([0,0,0]))
            self.rigid_num_particles = 0
        else:
            self.rigid = rigid
            self.rigid_num_particles = int(self.rigid.num_particles)
        self.offset = fluid.original_positions
        self.h = fluid.h
            
        self.max_num_particles = int(self.fluid.num_particles + self.rigid_num_particles)
        print("Max number of particles", self.max_num_particles)
        
        self.grid_x = int(width / fluid.grid_size_x) + 1
        self.grid_y = int(height / fluid.grid_size_y) + 1
        self.grid_z = int(depth / fluid.grid_size_z) + 1
        self.grid_size_x = 2 * width / (self.grid_x - 1)
        self.grid_size_y = 2 * height / (self.grid_y - 1)
        self.grid_size_z = 2 * depth / (self.grid_z - 1)
        
        self.idx_to_grid = ti.Vector.field(3, dtype=ti.i32, shape=(self.max_num_particles,))
        
        self.grid = ti.field(dtype=ti.i32)
        self.grid_num = ti.field(dtype=ti.i32)
        ti.root.dense(ti.ijk, (self.grid_x, self.grid_y, self.grid_z)).dynamic(ti.l, 1024).place(self.grid)
        ti.root.dense(ti.ijk, (self.grid_x, self.grid_y, self.grid_z)).place(self.grid_num)
        
        self.neighbour = ti.field(dtype=ti.i32)
        ti.root.dense(ti.i, self.max_num_particles).dynamic(ti.j, 2048).place(self.neighbour)
        self.neighbour_num = ti.field(dtype=ti.i32, shape=self.max_num_particles)
        
        if rigid is not None:
            self.rigid_positions = ti.Vector.field(3, dtype=ti.f32, shape=int(self.rigid_num_particles))
            self.rigid_velocities = ti.Vector.field(3, dtype=ti.f32, shape=int(self.rigid_num_particles))
            self.rigid_volumes = ti.field(dtype=ti.f32, shape=int(self.rigid_num_particles))
            self.rigid_masses = ti.field(dtype=ti.f32, shape=int(self.rigid_num_particles))
        else:
            self.rigid_positions = ti.Vector.field(3, dtype=ti.f32, shape=1)
            self.rigid_velocities = ti.Vector.field(3, dtype=ti.f32, shape=1)
            self.rigid_volumes = ti.field(dtype=ti.f32, shape=1)
            self.rigid_masses = ti.field(dtype=ti.f32, shape=1)
        self.is_fluid = ti.field(dtype=ti.i32, shape=self.max_num_particles)
        # self.update()
        
        min_x, min_y, min_z = self.domain_size + self.offset
        max_x, max_y, max_z = - self.domain_size + self.offset
        print("Boundary: ", min_x, min_y, min_z, max_x, max_y, max_z)
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

        for i in range(self.max_num_particles):
            self.neighbour_num[int(i)] = 0
            self.neighbour[int(i)].deactivate()
        
        for i in range(self.fluid.num_particles):
            self.is_fluid[i] = 1
        for i in range(self.rigid_num_particles):
            self.is_fluid[i+self.fluid.num_particles] = 0

    def get_rigid_pos(self):
        if self.rigid_num_particles == 0:
            return
        
        state = self.rigid.get_states()
        voxel = self.rigid.voxel
        self.rigid_positions_np = np.zeros((self.rigid_num_particles, 3), dtype=np.float32)
        transform = np.vstack((np.hstack((state["orientation"], state["position"].reshape(-1, 1))), [0, 0, 0, 1]))
        pos = transform @ np.hstack((voxel, np.ones((self.rigid_num_particles, 1)))).T
        self.rigid_positions_np = pos[:3].T
        self.rigid_velocities_np = state["velocity"] + np.cross(state["angular_velocity"], pos[:3].T - state["position"])
        # max_x, max_y, max_z = np.max(self.rigid_positions_np, axis=0)
        # min_x, min_y, min_z = np.min(self.rigid_positions_np, axis=0)
        # print(f"Rigid max_x: {max_x}, max_y: {max_y}, max_z: {max_z}")
        # print(f"Rigid min_x: {min_x}, min_y: {min_y}, min_z: {min_z}")
        self.rigid_positions.from_numpy(self.rigid_positions_np)
        self.rigid_velocities.from_numpy(self.rigid_velocities_np)
        
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
        
        for p_i in range(self.rigid_num_particles):
            pos = self.rigid_positions[p_i] - self.offset + self.domain_size
            x_id = int(pos[0] / self.grid_size_x)
            y_id = int(pos[1] / self.grid_size_y)
            z_id = int(pos[2] / self.grid_size_z)
            self.grid[x_id, y_id, z_id].append(p_i+self.fluid.num_particles)
            self.grid_num[x_id, y_id, z_id] += 1
            self.idx_to_grid[p_i+self.fluid.num_particles] = ti.Vector([x_id, y_id, z_id])

    @ti.func
    def update_neighbour(self):
        for p_i in range(self.fluid.num_particles):
            grid_idx = self.idx_to_grid[p_i]
            
            for offset in ti.grouped(ti.ndrange((-2, 3), (-2, 3), (-2, 3))):
                neighbor_idx = grid_idx + offset
                if 0 <= neighbor_idx[0] < self.grid_x and 0 <= neighbor_idx[1] < self.grid_y and 0 <= neighbor_idx[2] < self.grid_z:
                    for j in range(self.grid_num[neighbor_idx]):
                        p_j = self.grid[neighbor_idx,j]
                        r_len = 0.0
                        if self.is_fluid[p_j]:
                            r_len = (self.fluid.positions[p_j] - self.fluid.positions[p_i]).norm()
                        else:
                            r_len = (self.rigid_positions[p_j - self.fluid.num_particles] - self.fluid.positions[p_i]).norm()
                            
                        if r_len <= self.fluid.h:
                            self.neighbour[p_i].append(p_j)
                            self.neighbour_num[p_i] += 1
                            # print(self.is_fluid[p_j])
        
        for p_i in range(self.rigid_num_particles):
            grid_idx = self.idx_to_grid[p_i+self.fluid.num_particles]
            
            for offset in ti.grouped(ti.ndrange((-2, 3), (-2, 3), (-2, 3))):
                neighbor_idx = grid_idx + offset
                if 0 <= neighbor_idx[0] < self.grid_x and 0 <= neighbor_idx[1] < self.grid_y and 0 <= neighbor_idx[2] < self.grid_z:
                    for j in range(self.grid_num[neighbor_idx]):
                        p_j = self.grid[neighbor_idx,j]
                        r_len = 0.0
                        if self.is_fluid[p_j]:
                            r_len = (self.fluid.positions[p_j] - self.rigid_positions[p_i]).norm()
                        else:
                            r_len = (self.rigid_positions[p_j - self.fluid.num_particles] - self.rigid_positions[p_i]).norm()

                        if r_len <= self.fluid.h:
                            self.neighbour[p_i+self.fluid.num_particles].append(p_j)
                            self.neighbour_num[p_i+self.fluid.num_particles] += 1
        # avg_volume = 0.0
        # for i in range(self.rigid_num_particles):
        #     avg_volume += self.rigid_volumes[i]
        # avg_volume /= self.rigid_num_particles
        
    def positions_to_ply(self, path):
        self.fluid.positions_to_ply(os.path.join(path, "fluid.ply"))
        if self.rigid_num_particles > 0:
            rigid_positions = self.rigid_positions_np
            write_ply(rigid_positions, os.path.join(path, "rigid.ply"))
        # self.rigid.positions_to_ply(path)
    
    def save_mesh(self, path):
        # Create a mesh for the container itself
        min_x, min_y, min_z = self.domain_size + self.offset
        max_x, max_y, max_z = - self.domain_size + self.offset
        self.domain_size *= 2
        plane1 = trimesh.creation.box(extents=[self.domain_size[0], self.domain_size[1], 0.0001], transform=trimesh.transformations.translation_matrix([0.0, 0.0, min_z]))
        plane2 = trimesh.creation.box(extents=[self.domain_size[0], self.domain_size[1], 0.0001], transform=trimesh.transformations.translation_matrix([0.0, 0.0, max_z]))
        plane3 = trimesh.creation.box(extents=[self.domain_size[0], 0.0001, self.domain_size[2]], transform=trimesh.transformations.translation_matrix([0.0, min_y, 0.0]))
        plane4 = trimesh.creation.box(extents=[self.domain_size[0], 0.0001, self.domain_size[2]], transform=trimesh.transformations.translation_matrix([0.0, max_y, 0.0]))
        plane5 = trimesh.creation.box(extents=[0.0001, self.domain_size[1], self.domain_size[2]], transform=trimesh.transformations.translation_matrix([min_x, 0.0, 0.0]))
        plane6 = trimesh.creation.box(extents=[0.0001, self.domain_size[1], self.domain_size[2]], transform=trimesh.transformations.translation_matrix([max_x, 0.0, 0.0]))
        container_mesh = trimesh.util.concatenate([plane1, plane2, plane3, plane4, plane5, plane6])
        self.domain_size /= 2
        container_mesh.export(path)