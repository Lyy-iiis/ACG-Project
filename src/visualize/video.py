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
            if filename == 'output.png':
                filename = os.path.join(file_dir, filename)
                images.append(imageio.imread(os.path.join(images_dir, filename)))
    imageio.mimsave(output_path, images, fps=30, macro_block_size=8)
    
def convert_mp4_to_gif(input_file, output_file):
    from moviepy.editor import VideoFileClip    
    clip = VideoFileClip(input_file)
    clip.write_gif(output_file)