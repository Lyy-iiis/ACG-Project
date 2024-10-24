import open3d as o3d
import numpy as np
import tqdm

def visualize_ply(file_path):
    # Load the PLY file
    import matplotlib.pyplot as plt

    # Load the PLY file
    ply_data = o3d.io.read_point_cloud(file_path)
    points = np.asarray(ply_data.points)

    # Plot the points using matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    file_path = file_path.split("/")[:-1]
    file_path = "/".join(file_path)
    plt.savefig(file_path + "/output_pc.png")
    plt.close()

def visualize(output_dir, frame):
    for i in tqdm.tqdm(range(frame)):
        file_path = f"{output_dir}/{i}/output.ply"
        visualize_ply(file_path)