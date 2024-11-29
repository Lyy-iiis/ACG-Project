import taichi as ti

ti.init(arch=ti.cpu)

n = 128
quad_size = 1.0 / n
dt = 4e-2 / n
substeps = int(1 / 60 // dt)

# Default simulation parameters
gravity = ti.Vector([0.0, -9.8, 0.0])
spring_Y = ti.field(float, shape=())
dashpot_damping = ti.field(float, shape=())
drag_damping = ti.field(float, shape=())
ball_radius = 0.3
ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
ball_center[0] = [0, 0, 0]

# Initialize the parameters
spring_Y[None] = 2e4
dashpot_damping[None] = 1e4
drag_damping[None] = 1

# Mass point fields
x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))

# Mesh-related fields
num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

bending_springs = False

@ti.kernel
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1
    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5 + random_offset[0], 0.6,
            j * quad_size - 0.5 + random_offset[1]
        ]
        v[i, j] = [0, 0, 0]


@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # First triangle of the square
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # Second triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = (0.51, 0.13, 0.13)
        else:
            colors[i * n + j] = (0.7, 0, 0)


initialize_mesh_indices()

spring_offsets = []
if bending_springs:
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i, j) != (0, 0):
                spring_offsets.append(ti.Vector([i, j]))
else:
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                spring_offsets.append(ti.Vector([i, j]))


@ti.kernel
def substep(g_x: float, g_y: float, g_z: float, spring_y: float, dashpot_d: float, drag_d: float):
    for i in ti.grouped(x):
        v[i] += dt * ti.Vector([g_x, g_y, g_z])

    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                original_dist = quad_size * float(i - j).norm()
                # Spring force
                force += -spring_y * d * (current_dist / original_dist - 1)
                # Dashpot damping
                force += -v_ij.dot(d) * d * dashpot_d * quad_size

        v[i] += force * dt

    for i in ti.grouped(x):
        v[i] *= ti.exp(-drag_d * dt)
        offset_to_center = x[i] - ball_center[0]
        if offset_to_center.norm() <= ball_radius:
            # Velocity projection
            normal = offset_to_center.normalized()
            v[i] -= min(v[i].dot(normal), 0) * normal
        x[i] += dt * v[i]


@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]


# GUI and simulation setup
window = ti.ui.Window("Taichi Cloth Simulation with Controls", (1024, 1024))
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
gui = window.get_gui()

running = ti.field(int, shape=())
running[None] = 1  # 1 means running, 0 means paused

initialize_mass_points()

current_t = 0.0

while window.running:
    with gui.sub_window("Controls", 0.02, 0.02, 0.3, 0.5):
        gui.text("Simulation Parameters")
        gravity[0] = gui.slider_float("Gravity X", gravity[0], -20.0, 20.0)
        gravity[1] = gui.slider_float("Gravity Y", gravity[1], -20.0, 20.0)
        gravity[2] = gui.slider_float("Gravity Z", gravity[2], -20.0, 20.0)
        spring_Y[None] = gui.slider_float("Spring Stiffness", spring_Y[None], 1e3, 4e4)
        dashpot_damping[None] = gui.slider_float("Dashpot Damping", dashpot_damping[None], 1e3, 2e4)
        drag_damping[None] = gui.slider_float("Drag Damping", drag_damping[None], 0.1, 5.0)

        gui.text("Controls")
        if gui.button("Start/Pause"):
            running[None] = 1 - running[None]
        if gui.button("Reset"):
            initialize_mass_points()
            current_t = 0.0

    if running[None]:
        for i in range(substeps):
            substep(*(gravity.to_list()), spring_Y[None], dashpot_damping[None], drag_damping[None])
            current_t += dt
        update_vertices()

    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices, indices=indices, per_vertex_color=colors, two_sided=True)

    # Draw a smaller ball to avoid visual penetration
    scene.particles(ball_center, radius=ball_radius * 0.8, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()
