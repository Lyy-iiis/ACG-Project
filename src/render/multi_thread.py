import os
import multiprocessing as mp
from tqdm import tqdm
# from contextlib import contextmanager
from src.material import utils
from src.render import render
import sys
            
def process_frame(Renderer: render.Render, output_dir, i):
    mesh = utils.get_rigid_from_mesh(f'{output_dir}/{i}/output.obj')
    Renderer.render_fluid_mesh(mesh, f'{output_dir}/{i}/output.png')
    
def worker(Renderer, output_dir, frame):
    try:
        process_frame(Renderer, output_dir, frame)
    except Exception as e:
        print(f"failed to process {frame}")
        print(e)
    return 1 # return 1 to indicate success

def process(output_dir, frame_num):
    # Using a pool of workers to process the images
    pool = mp.Pool(4)

    # Progress bar setup
    pbar = tqdm(total=frame_num)

    # Update progress bar in callback
    def update_pbar(result):
        pbar.update(1)
    Renderer = render.Render()
    for i in range(frame_num):
        pool.apply_async(worker, args=(Renderer, output_dir, i,), callback=update_pbar)
    pool.close()
    pool.join()
    pbar.close()
    pool.join()