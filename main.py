import taichi as ti
import src.material
import src.material.geometry
from src.render import render, video
from src.material import rigid, fluid, cloth, utils, container
import src.render.utils
import matplotlib.pyplot as plt
import numpy as np
import os

object_name = 'bunny'
device = ti.gpu # Set to ti.cpu when debugging
output_dir = 'output'
Dt = 3e-5
Frame = 1
demo = True
substeps = int(1 / 60 // Dt)

def test_rigid():
    # ti.init(arch=device)
    
    Renderer = render.Render()
    print("Starting main function")
    
    mesh = utils.get_rigid_from_mesh(f'assets/{object_name}.obj')
    print("Mesh loaded successfully")
    
    Rigid_1 = rigid.RigidBody(mesh=mesh, position=np.array([0,0,-4]))
    Rigid_2 = rigid.RigidBody("Ball", radius=0.3, position=np.array([1,0,-4]))
    print("Rigid body created successfully")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    force = ti.Vector([0.5,0.5,0.5]) # don't apply too big force !!!
    mesh_1 = src.render.utils.trimesh_to_blender_object(
            utils.mesh(Rigid_1.vertices, Rigid_1.faces), 
            object_name=object_name)
    mesh_2 = src.render.utils.trimesh_to_blender_object(
            utils.mesh(Rigid_2.vertices, Rigid_2.faces), 
            object_name="Ball")
    
    for i in range(100):
        Rigid_1.apply_external_force(force, ti.Vector([0.0, 0.0, 0.0]))
        Rigid_2.apply_external_force(-force, ti.Vector([0.0, 0.0, 0.0]))
        Rigid_1.update(0.01)
        Rigid_2.update(0.01)
        print(f"Frame {i}")
        Renderer.render_rigid_body([mesh_1,mesh_2], [Rigid_1,Rigid_2], f'{output_dir}/output_{i}.png')
        
    video.create_video(output_dir, 'output.mp4')
    
def test_fluid():
    Renderer = render.Render()
    print("Starting main function")
    
    mesh = utils.get_rigid_from_mesh(f'assets/{object_name}.obj')
    box_size = [0.25, 0.25, 0.25]
    # mesh = src.material.geometry.Box(box_size, [0.0, 0.0, 0.0])
    # print(mesh.vertices)
    print("Mesh loaded successfully")
    # resolution = 0.04
    # mass = 0.5 ** 3 * 1000
    # particle_mass = 4 * resolution ** 3 * np.pi * 1000 / (3 * 100)
    # particle_num = int(mass / particle_mass)
    # print(f"Particle number: {particle_num}")
    # print(f"Particle mass: {particle_mass}")
    Fluid = fluid.Fluid([0.5, 0.5, 0.5], mesh, position=np.array([0,0,-4]))
    print("Fluid created successfully")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Container = container.Container(1.0, 1.0, 1.0, Fluid)

    if demo:
        for i in range(Frame):
            print(f"Frame {i}")
            if not os.path.exists(f'{output_dir}/{i}'):
                os.makedirs(f'{output_dir}/{i}')
            # Container.update()
            Fluid.step()
            Fluid.positions_to_ply(f'{output_dir}/{i}/output_{i}.ply')
        
        os.system(f"python3 src/surface.py --input_dir {output_dir}")
        
        for i in range(Frame):
            mesh = utils.get_rigid_from_mesh(f'{output_dir}/{i}/output_{i}.obj')
            Renderer.render_fluid_mesh(mesh, f'{output_dir}/{i}/output_{i}.png')
    else:
        Renderer.add_fluid(Fluid)
        for i in range(Frame):
            print(f"Frame {i}")
            if not os.path.exists(f'{output_dir}/{i}'):
                os.makedirs(f'{output_dir}/{i}')
            # Container.update()
            Fluid.step()
            Renderer.render_fluid(Fluid, f'{output_dir}/{i}/output_{i}.png')
            
    video.create_video(output_dir, 'output.mp4')

def test_cloth1():
    Renderer = render.Render()
    print("Starting main function")

    mesh = utils.get_rigid_from_mesh(f'assets/{object_name}.obj')
    print("Mesh loaded successfully")

    # Rigid_1 = rigid.RigidBody(mesh=mesh, position=np.array([0, 0, -8]))

    Cloth = cloth.Cloth(particle_mass=0.1, initial_position=np.array([0, 1, -8]))
    print("Cloth created successfully")
    
    flat_positions = ti.Vector.field(3, dtype=ti.f32, shape=(Cloth.num_particles,))


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(100):
        Cloth.get_flat_positions(flat_positions)
        # mesh_rigid = src.render.utils.trimesh_to_blender_object(
        #         utils.mesh(Rigid_1.vertices, Rigid_1.faces), 
        #         object_name="Rigid")
        mesh_cloth = src.render.utils.trimesh_to_blender_object(
                utils.mesh(flat_positions, Cloth.faces),
                object_name="ClothMesh")

        for _ in range(substeps):  
            Cloth.substep()
            # Rigid_1.update(Dt)

            # Detecting collisions between cloth and rigid bodies
            # Cloth.collision_detection(Rigid_1, 0.01)
            
            # Update the state of cloth and rigidbody
            # Cloth.update(0.001)
            # Rigid_1.update(0.01)

        print(f"Frame {i}")
        # Renderer.render_cloth1([mesh_cloth, mesh_rigid], f'{output_dir}/output_{i}.png')
        Renderer.render_cloth1(mesh_cloth, f'{output_dir}/output_{i}.png')
        # Renderer.render_rigid_body([mesh_rigid], [Rigid_1], f'{output_dir}/output_{i}.png')


    video.create_video(output_dir, 'output.mp4')

    
 
def main():
    ti.init(arch=device)
    
    # test_rigid()
    test_fluid()
    # test_cloth1()
    
if __name__ == '__main__':
    main()