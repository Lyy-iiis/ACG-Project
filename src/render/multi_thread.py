import os
import multiprocessing as mp
from tqdm import tqdm
# from contextlib import contextmanager
from src.material import utils
from src.render import render
import sys
import math
            
def process_frame(Renderer: render.Render, output_dir, i, is_coupled):
    if is_coupled:
        rigid_mesh = utils.get_rigid_from_mesh(f'{output_dir}/{i}/rigid.obj')
        fluid_mesh = utils.get_rigid_from_mesh(f'{output_dir}/{i}/fluid.obj')
        container_mesh = utils.get_rigid_from_mesh(f'{output_dir}/0/container.obj')
        Renderer.render_coupled_fluid_rigid(fluid_mesh, rigid_mesh, container_mesh, f'{output_dir}/{i}/output.png')
    else:
        fluid_mesh = utils.get_rigid_from_mesh(f'{output_dir}/{i}/fluid.obj')
        Renderer.render_fluid(fluid_mesh, f'{output_dir}/{i}/output.png')
    
def worker(Renderer, output_dir, frame, is_coupled):
    try:
        process_frame(Renderer, output_dir, frame, is_coupled)
    except Exception as e:
        print(f"failed to process {frame}")
        print(e)
    return 1 # return 1 to indicate success

def process(output_dir, frame_num, is_coupled=False):
    # Using a pool of workers to process the images
    pool = mp.Pool(4)

    # Progress bar setup
    pbar = tqdm(total=frame_num)

    # Update progress bar in callback
    def update_pbar(result):
        pbar.update(1)
    Renderer = render.Render(camera_location=[-3+0.87-0.4-0.433*2, 3.8-0.9+0.5*2-0.2, 1-1.5+0.75*2], camera_rotation=[math.radians(-30), math.radians(-30), 0])
    
    for i in range(frame_num):
        pool.apply_async(worker, args=(Renderer, output_dir, i, is_coupled), callback=update_pbar)

    pool.close()
    pool.join()
    pbar.close()
    pool.join()