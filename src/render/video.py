import imageio
import os

def create_video(images_dir, output_path):
    images = []
    for filename in sorted(os.listdir(images_dir)):
        if filename.endswith('.png'):
            images.append(imageio.imread(os.path.join(images_dir, filename)))
    imageio.mimsave(output_path, images, fps=30)