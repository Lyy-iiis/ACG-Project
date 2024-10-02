import taichi as ti
from render import render, video
from material import rigid
import bpy
import numpy as np

object_name = 'bunny'
device = ti.gpu

def main():
    ti.init(arch=device)
    
    render.init_render()
    print("Starting main function")
    
    mesh = rigid.get_rigid_from_mesh(f'assets/{object_name}.obj')
    print("Mesh loaded successfully")
    
    Rigid = rigid.RigidBody(mesh, position=np.array([0,0,-4]))
    print("Rigid body created successfully")
    
    force = ti.Vector([0.1,0.1,0.1]) # don't apply too big force !!!
    mesh = render.trimesh_to_blender_object(Rigid.mesh(), object_name=object_name)
    for i in range(100):
        Rigid.apply_force(force, ti.Vector([0.0, 0.0, 0.0]))
        Rigid.update(0.01)
        print(f"Frame {i}")
        render.render_rigid_body(mesh, Rigid, f'output/output_{i}.png')
        
    video.create_video('output', 'output.mp4')
    
if __name__ == '__main__':
    main()