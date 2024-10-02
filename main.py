import taichi as ti
from src.render import render, video
from src.material import rigid, utils
import numpy as np
import os

object_name = 'Bunny'
device = ti.gpu
output_dir = 'output'

def main():
    ti.init(arch=device)
    
    render.init_render()
    print("Starting main function")
    
    mesh = utils.get_rigid_from_mesh(f'assets/{object_name}.obj')
    print("Mesh loaded successfully")
    
    Rigid = rigid.RigidBody(mesh, position=np.array([0,0,-4]))
    print("Rigid body created successfully")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    force = ti.Vector([0.1,0.1,0.1]) # don't apply too big force !!!
    mesh = render.trimesh_to_blender_object(utils.mesh(Rigid.vertices, Rigid.faces), 
                                            object_name=object_name)
    for i in range(100):
        Rigid.apply_force(force, ti.Vector([0.0, 0.0, 0.0]))
        Rigid.update(0.01)
        print(f"Frame {i}")
        render.render_rigid_body(mesh, Rigid, f'{output_dir}/output_{i}.png')
        
    video.create_video(output_dir, 'output.mp4')
    
if __name__ == '__main__':
    main()