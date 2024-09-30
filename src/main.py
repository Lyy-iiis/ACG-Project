# import taichi as ti
from render import render
from material import rigid
import bpy


def main():
    render.init_render()
    print("Starting main function")
    mesh = rigid.get_rigid_from_mesh('assets/bunny.obj')
    mesh = render.trimesh_to_blender_object(mesh)
    
    print("Mesh loaded successfully")
    render.render_mesh(mesh, 'output.png')
    print("Render completed successfully")
    
if __name__ == '__main__':
    main()