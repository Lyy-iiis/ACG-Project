import imageio
import os
import re

def create_video(images_dir, output_path):
    images = []
    def sort_key(filename):
        match = re.search(r'output_(\d+)', filename)
        return int(match.group(1)) if match else float('inf')
    for file_dir in sorted(os.listdir(images_dir), key=sort_key):
        for filename in sorted(os.listdir(os.path.join(images_dir, file_dir)), key=sort_key):
            if filename.endswith('.png') and filename.startswith('output_'):
                filename = os.path.join(file_dir, filename)
                images.append(imageio.imread(os.path.join(images_dir, filename)))
    imageio.mimsave(output_path, images, fps=30, macro_block_size=8)