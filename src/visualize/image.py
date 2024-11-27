import os
import shutil

def move_png_files(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    for filename in os.listdir(src_dir):
        if filename.endswith('.png'):
            src_file = os.path.join(src_dir, filename)
            dest_file = os.path.join(dest_dir, filename)
            shutil.copy(src_file, dest_file)
            print(f'Moved: {src_file} to {dest_file}')
    for root, path, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.png'):
                src_file = os.path.join(root, file)
                num = src_file.split('/')[-2]
                file_name = f'{num}_{file}'
                dest_file = os.path.join(dest_dir, file_name)
                # dest_file = os.path.join(dest_dir, os.path.relpath(src_file, src_dir))
                os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                shutil.copy(src_file, dest_file)
                print(f'Moved: {src_file} to {dest_file}')

# Example usage
source_directory = './output_1'
destination_directory = './output_1_png'
move_png_files(source_directory, destination_directory)