from src.material.fluid import *
from src.material.rigid import *
import os
from src.material.geometry import *

@ti.data_oriented
class Container:
    def __init__(self, width, height, depth, fluid: Fluid, rigid: RigidBody):
        self.domain_size = ti.Vector([width, height, depth])
        self.fluid = fluid
        self.rigid = rigid
        self.offset = fluid.original_positions
        self.h = fluid.h
        self.max_num_particles = self.fluid.num_particles
        self.max_num_particles = int(self.fluid.num_particles + self.rigid.num_particles)
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
        
        self.rigid_positions = ti.Vector.field(3, dtype=ti.f32, shape=int(self.rigid.num_particles))
        self.rigid_velocities = ti.Vector.field(3, dtype=ti.f32, shape=int(self.rigid.num_particles))
        self.rigid_volumes = ti.field(dtype=ti.f32, shape=int(self.rigid.num_particles))
        self.rigid_masses = ti.field(dtype=ti.f32, shape=int(self.rigid.num_particles))
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
        for i in range(self.rigid.num_particles):
            self.is_fluid[i+self.fluid.num_particles] = 0

    def get_rigid_pos(self):
        state = self.rigid.get_states()
        voxel = self.rigid.voxel
        self.rigid_positions_np = np.zeros((self.rigid.num_particles, 3), dtype=np.float32)
        transform = np.vstack((np.hstack((state["orientation"], state["position"].reshape(-1, 1))), [0, 0, 0, 1]))
        pos = transform @ np.hstack((voxel, np.ones((self.rigid.num_particles, 1)))).T
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
        
        for p_i in range(self.rigid.num_particles):
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
                            
        for p_i in range(self.rigid.num_particles):
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
      
    @ti.func
    def update_rigid_particle_volume(self):
        for i in range(self.rigid.num_particles):
            pos_i = self.rigid_positions[i]
            self.rigid_volumes[i] = 0.0
            for j in range(self.neighbour_num[i+self.fluid.num_particles]):
                p_j = self.neighbour[i+self.fluid.num_particles,j]
                if not self.is_fluid[p_j]:
                    pos_j = self.rigid_positions[p_j-self.fluid.num_particles]
                    R = pos_i - pos_j
                    R_mod = R.norm()
                    self.rigid_volumes[i] += self.fluid.kernel_func(R_mod)
            self.rigid_volumes[i] = 1.0 / self.rigid_volumes[i]
            self.rigid_masses[i] = self.rigid_volumes[i] * self.fluid.rest_density
        
        avg_volume = 0.0
        for i in range(self.rigid.num_particles):
            avg_volume += self.rigid_volumes[i]
        avg_volume /= self.rigid.num_particles
            
    
    @ti.func
    def compute_densities_and_pressures(self):
        for i in range(self.fluid.num_particles):
            self.fluid.densities[i] = 0.0
            self.fluid.pressures[i] = 0.0
            for j in range(self.neighbour_num[i]):
                p_j = self.neighbour[i,j]
                if self.is_fluid[p_j]:
                    self.fluid.compute_densities_and_pressures(i, p_j)
                else:
                    pos_i = self.fluid.positions[i]
                    pos_j = self.rigid_positions[p_j-self.fluid.num_particles]
                    R = pos_i - pos_j
                    R_mod = R.norm()
                    self.fluid.densities[i] += self.fluid.kernel_func(R_mod) * self.rigid_masses[p_j-self.fluid.num_particles]
                    
            self.fluid.densities[i] = ti.max(self.fluid.densities[i], self.fluid.rest_density)
            self.fluid.pressures[i] = self.fluid.stiffness * ((self.fluid.densities[i] / self.fluid.rest_density) ** 7 - 1)

        self.fluid.avg_density[None] = 0.0
        for i in range(self.fluid.num_particles):
            self.fluid.avg_density[None] += self.fluid.densities[i]
        self.fluid.avg_density[None] /= self.fluid.num_particles
    
    @ti.func
    def compute_forces_rigid(self, i, p_j):
        m_ij = self.rigid_masses[p_j-self.fluid.num_particles]
        v_xy = self.fluid.velocities[i] - self.rigid_velocities[p_j-self.fluid.num_particles]
        R = self.fluid.positions[i] - self.rigid_positions[p_j-self.fluid.num_particles]
        v_xy = ti.math.dot(v_xy, R)
        nabla_ij = self.fluid.kernel_grad(R)
        
        viscosity_force = 2 * 5 * self.fluid.viscosity * m_ij / self.fluid.densities[i] / (R.norm() ** 2 + 0.01 * self.fluid.h ** 2) * v_xy * nabla_ij / self.fluid.rest_density
        pressure_force = - m_ij * self.fluid.pressures[i] / (self.fluid.densities[i] ** 2) * nabla_ij
        force = pressure_force + viscosity_force
        self.fluid.forces[i] += force
        
        force_j = - force * self.fluid.mass[i] / self.rigid_masses[p_j-self.fluid.num_particles]
        pos_j = self.rigid_positions[p_j-self.fluid.num_particles]
        self.rigid.apply_internal_force(force_j, pos_j)
        
    @ti.func
    def compute_forces(self):
        for i in range(self.fluid.num_particles):
            self.fluid.forces[i] = ti.Vector([0.0, 0.0, 0.0])
            for j in range(self.neighbour_num[i]):
                p_j = self.neighbour[i,j]
                if p_j < self.fluid.num_particles:
                    self.fluid.compute_forces(i, p_j)
                else:
                    self.compute_forces_rigid(i, p_j)
                    
            self.fluid.forces[i] += self.fluid.gravity[None]
            self.fluid.forces[i] *= self.fluid.mass[i]
        
        # self.rigid.force[None] += self.rigid.mass * self.fluid.gravity[None]
    
    @ti.kernel
    def update(self):
        self.empty_grid()
        self.update_grid()
        self.update_neighbour()
        self.update_rigid_particle_volume()
        self.compute_densities_and_pressures()
        self.compute_forces()
        self.fluid.update_particles()
        self.enforce_domain_boundary()
        
    def step(self):
        self.get_rigid_pos()
        self.update()
        # self.rigid.update(self.fluid.time_step)
        
    def positions_to_ply(self, path):
        self.fluid.positions_to_ply(os.path.join(path, "fluid.ply"))
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