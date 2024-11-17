import taichi as ti
# from material.fluid import basefluid, WCSPH
import src.material
import src.material.geometry
from src.render import render, multi_thread
from src.material import rigid, cloth, utils
from src.material.fluid import basefluid, WCSPH, DFSPH
from src.material.container import base_container, WCSPH_container, DFSPH_container
from src.visualize import visualizer, video
import src.render.utils
from tqdm import tqdm
import numpy as np
import os

object_name = 'bunny'
device = ti.gpu # Set to ti.cpu when debugging
output_dir = 'output'
Dt = 3e-5
Frame = 300
demo = True
substeps = int(1 / 60 // Dt)

def test_rigid():
    Renderer = render.Render()
    
    mesh = utils.get_rigid_from_mesh(f'assets/{object_name}.obj')
    print("Mesh loaded successfully")
    
    Rigid_1 = rigid.RigidBody(mesh=mesh, position=np.array([0,0,-4]))
    Rigid_2 = rigid.RigidBody("Ball", radius=0.3, position=np.array([1,0,-4]))
    print("Rigid body created successfully")
        
    force = ti.Vector([0.5,0.5,0.5]) # don't apply too big force !!!
    mesh_1 = src.render.utils.trimesh_to_blender_object(
            utils.mesh(Rigid_1.vertices, Rigid_1.faces), 
            object_name=object_name)
    mesh_2 = src.render.utils.trimesh_to_blender_object(
            utils.mesh(Rigid_2.vertices, Rigid_2.faces), 
            object_name="Ball")
    
    for i in range(Frame):
        if not os.path.exists(f'{output_dir}/{i}'):
            os.makedirs(f'{output_dir}/{i}')
        Rigid_1.apply_external_force(force, ti.Vector([0.0, 0.0, 0.0]))
        Rigid_2.apply_external_force(-force, ti.Vector([0.0, 0.0, 0.0]))
        Rigid_1.update(0.01)
        Rigid_2.update(0.01)
        print(f"Frame {i}")
        Renderer.render_rigid_body([mesh_1,mesh_2], [Rigid_1,Rigid_2], f'{output_dir}/{i}/output.png')
        
    video.create_video(output_dir, 'output.mp4')
    
def test_fluid():
    Renderer = render.Render() # Don't remove this line even if it is not used
    
    # mesh = utils.get_rigid_from_mesh(f'assets/{object_name}.obj')
    box_size = [0.6, 1.6, 0.4]
    mesh = src.material.geometry.Box(extents=box_size, center=[0.5, 0.0, 0.0])
    print(mesh.vertices)
    print("Mesh loaded successfully")
    
    Fluid = DFSPH.DFSPH(mesh, position=np.array([0.5,0,-6]))
    Container = DFSPH_container.DFSPHContainer(1.2, 1, 0.3, Fluid, None)
    
    # Fluid = WCSPH.WCSPH(mesh, position=np.array([0.5,0,-6]))
    # Container = WCSPH_container.WCSPHContainer(1.2, 1, 0.3, Fluid, None)

    substeps = int(1 / (Fluid.fps * Fluid.time_step))
    Container.prepare()
    for i in range(Frame):
        if not os.path.exists(f'{output_dir}/{i}'):
            os.makedirs(f'{output_dir}/{i}')
        Container.positions_to_ply(f'{output_dir}/{i}')
        for _ in tqdm(range(substeps), desc=f"Frame {i}, Avg pos {Fluid.avg_position.to_numpy()[1]:.2f}, Avg density {Fluid.avg_density.to_numpy():.2f}"):
            Container.step()
    
    if not os.path.exists(f'{output_dir}/{Frame}'):
        os.makedirs(f'{output_dir}/{Frame}')
    Container.save_mesh(f'{output_dir}/{Frame}/container.obj')
    
    print("Visualizing the fluid") 
    if demo:
        os.system(f"python3 src/visualize/surface.py --input_dir {output_dir} --frame {Frame+1}")
        multi_thread.process(output_dir, Frame)
    else:
        visualizer.visualize(output_dir, Frame)
            
    video.create_video(output_dir, 'output.mp4')

def test_cloth1():
    Renderer = render.Render()

    mesh = utils.get_rigid_from_mesh(f'assets/{object_name}.obj')
    print("Mesh loaded successfully")

    # Rigid_1 = rigid.RigidBody(mesh=mesh, position=np.array([0, 0, -8]))

    Cloth = cloth.Cloth(particle_mass=0.1, initial_position=np.array([0, 1, -8]))
    print("Cloth created successfully")
    
    flat_positions = ti.Vector.field(3, dtype=ti.f32, shape=(Cloth.num_particles,))

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
        Renderer.render_cloth1(mesh_cloth, f'{output_dir}/output.png')
        # Renderer.render_rigid_body([mesh_rigid], [Rigid_1], f'{output_dir}/output_{i}.png')


    video.create_video(output_dir, 'output.mp4')

def test_coupling():    
    Renderer = render.Render() # Don't remove this line even if it is not used
    
    mesh1 = utils.get_rigid_from_mesh(f'assets/{object_name}.obj')
    # box_size = [1.2, 0.8, 0.5]
    box_size = [0.4, 0.4, 0.4]
    mesh = src.material.geometry.Box(extents=box_size, center=[0.0, 0.0, 0.0])
    print("Mesh loaded successfully")
    
    Rigid = rigid.RigidBody(mesh=mesh1, position=np.array([0.5,-0.5,-5],dtype=np.float32))
    Fluid = basefluid.Fluid(mesh, position=np.array([0,0.55,-5],dtype=np.float32))
    Container = base_container.Container(1.2, 1.5, 0.5, Fluid, Rigid)

    substeps = int(1 / (Fluid.fps * Fluid.time_step))
    for i in range(Frame):
        if not os.path.exists(f'{output_dir}/{i}'):
            os.makedirs(f'{output_dir}/{i}')
        for _ in tqdm(range(substeps), desc=f"Frame {i}, Avg pos {Fluid.avg_position.to_numpy()[1]:.2f}, Avg density {Fluid.avg_density.to_numpy():.2f}"):
            # Fluid.step()
            Container.step()
        if i == 0:
            Container.save_mesh(f'{output_dir}/{i}/container.obj')
        Container.positions_to_ply(f'{output_dir}/{i}')
    
    print("Visualizing the fluid") 
    if demo:
        os.system(f"python3 src/visualize/surface.py --input_dir {output_dir} --frame {Frame}")
        multi_thread.process(output_dir, Frame, is_coupled=True)
    else:
        visualizer.visualize(output_dir, Frame)
            
    video.create_video(output_dir, 'output.mp4')
    
def main():
    print("Starting main function")
    ti.init(arch=device,device_memory_fraction=0.95,debug=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # test_rigid()
    test_fluid()
    # test_cloth1()
    # test_coupling()
    
if __name__ == '__main__':
    main()