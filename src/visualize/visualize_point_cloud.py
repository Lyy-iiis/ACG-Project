import open3d as o3d
import numpy as np

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
    plt.savefig(file_path + ".png")

if __name__ == "__main__":
    for i in range(20):
        print(i)
        file_path = f"output/{i}/output_{i}.ply"  # Replace with your actual file path
        visualize_ply(file_path)