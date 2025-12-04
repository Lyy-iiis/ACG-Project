import taichi as ti
import numpy as np
from taichi.linalg import SparseMatrixBuilder
from enum import Enum
import os
import openvdb as vdb
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg

ti.init(arch=ti.cpu)

class InterpolationMethod(Enum):
    LINEAR = 0
    MONOTONIC_CUBIC = 1
   
class AdvectionMethod(Enum):
    SEMI_LAGRANGE = 0
    MAC_CORMACK = 1

class EmitterPosition(Enum):
    E_TOP = 0
    E_BOTTOM = 1

N = 32
ratio = [1.0, 2.0, 1.0]

Nx = int(ratio[0] * N)
Ny = int(ratio[1] * N)
Nz = int(ratio[2] * N)
SOURCE_SIZE_X = Nx // 4
SOURCE_SIZE_Y = Ny // 20
SOURCE_SIZE_Z = Nz // 4
SOURCE_Y_MARGIN = Ny // 20
SIZE = Nx * Ny * Nz

DT = np.float64(0.02)
VOXEL_SIZE = np.float64(1.0)
INIT_DENSITY = np.float64(1.0)
INIT_VELOCITY = np.float64(80.0)
VORT_EPS = np.float64(0.25)  # 0.0
ALPHA = np.float64(9.8)  # 0.98
BETA = np.float64(15.0)  # 1.5
T_AMP = np.float64(5.0)
T_AMBIENT = np.float64(50.0)
EMIT_DURATION = np.float64(2.0)
FINISH_TIME = np.float64(6.0)
INTERPOLATION_METHOD = InterpolationMethod.LINEAR
ADVECTION_METHOD = AdvectionMethod.MAC_CORMACK
EMITTER_POS = EmitterPosition.E_TOP

A_global = None
b_global = None
x_global = None

tolerance = 1e-8


# Fields Definition
density = ti.field(dtype=ti.f64, shape=(Nx, Ny, Nz))
density0 = ti.field(dtype=ti.f64, shape=(Nx, Ny, Nz))
temperature = ti.field(dtype=ti.f64, shape=(Nx, Ny, Nz))
temperature0 = ti.field(dtype=ti.f64, shape=(Nx, Ny, Nz))

pressure = ti.field(dtype=ti.f64, shape=(Nx, Ny, Nz))

u = ti.field(dtype=ti.f64, shape=(Nx + 1, Ny, Nz))
u0 = ti.field(dtype=ti.f64, shape=(Nx + 1, Ny, Nz))
v = ti.field(dtype=ti.f64, shape=(Nx, Ny + 1, Nz))
v0 = ti.field(dtype=ti.f64, shape=(Nx, Ny + 1, Nz))
w = ti.field(dtype=ti.f64, shape=(Nx, Ny, Nz + 1))
w0 = ti.field(dtype=ti.f64, shape=(Nx, Ny, Nz + 1))

avg_u = ti.field(dtype=ti.f64, shape=(Nx, Ny, Nz))
avg_v = ti.field(dtype=ti.f64, shape=(Nx, Ny, Nz))
avg_w = ti.field(dtype=ti.f64, shape=(Nx, Ny, Nz))

omg_x = ti.field(dtype=ti.f64, shape=(Nx, Ny, Nz))
omg_y = ti.field(dtype=ti.f64, shape=(Nx, Ny, Nz))
omg_z = ti.field(dtype=ti.f64, shape=(Nx, Ny, Nz))
vort = ti.field(dtype=ti.f64, shape=(Nx, Ny, Nz))

fx = ti.field(dtype=ti.f64, shape=(Nx, Ny, Nz))
fy = ti.field(dtype=ti.f64, shape=(Nx, Ny, Nz))
fz = ti.field(dtype=ti.f64, shape=(Nx, Ny, Nz))

time = ti.field(dtype=ti.f64, shape=())
time[None] = 0.0


@ti.kernel
def initialize():
    for i, j, k in temperature:
        rand_val = ti.random(ti.f64) * T_AMP
        temperature[i, j, k] = (j / Ny) * T_AMP + T_AMBIENT
        # temperature[i, j, k] = (j / Ny) * T_AMP + rand_val + T_AMBIENT
        density[i, j, k] = 0.0
    for I in ti.grouped(u):
        u[I] = 0.0
    for I in ti.grouped(v):
        v[I] = 0.0
    for I in ti.grouped(w):
        w[I] = 0.0


@ti.kernel
def addSource():
    if EMITTER_POS == EmitterPosition.E_TOP:
        for k in range((Nz - SOURCE_SIZE_Z) // 2, (Nz + SOURCE_SIZE_Z) // 2):
            for j in range(SOURCE_Y_MARGIN, SOURCE_Y_MARGIN + SOURCE_SIZE_Y):
                for i in range((Nx - SOURCE_SIZE_X) // 2, (Nx + SOURCE_SIZE_X) // 2):
                    density[i, j, k] = INIT_DENSITY
    elif EMITTER_POS == EmitterPosition.E_BOTTOM:
        for k in range((Nz - SOURCE_SIZE_Z) // 2, (Nz + SOURCE_SIZE_Z) // 2):
            for j in range(Ny - SOURCE_Y_MARGIN - SOURCE_SIZE_Y, Ny - SOURCE_Y_MARGIN):
                for i in range((Nx - SOURCE_SIZE_X) // 2, (Nx + SOURCE_SIZE_X) // 2):
                    density[i, j, k] = INIT_DENSITY


@ti.kernel
def setEmitterVelocity():
    if EMITTER_POS == EmitterPosition.E_TOP:
        for k in range((Nz - SOURCE_SIZE_Z) // 2, (Nz + SOURCE_SIZE_Z) // 2):
            for j in range(SOURCE_Y_MARGIN, SOURCE_Y_MARGIN + SOURCE_SIZE_Y):
                for i in range((Nx - SOURCE_SIZE_X) // 2, (Nx + SOURCE_SIZE_X) // 2):
                    v[i, j, k] = INIT_VELOCITY
                    v0[i, j, k] = v[i, j, k]
    elif EMITTER_POS == EmitterPosition.E_BOTTOM:
        for k in range((Nz - SOURCE_SIZE_Z) // 2, (Nz + SOURCE_SIZE_Z) // 2):
            for j in range(Ny - SOURCE_Y_MARGIN - SOURCE_SIZE_Y, Ny - SOURCE_Y_MARGIN + 1):
                for i in range((Nx - SOURCE_SIZE_X) // 2, (Nx + SOURCE_SIZE_X) // 2):
                    v[i, j, k] = -INIT_VELOCITY
                    v0[i, j, k] = v[i, j, k]


@ti.kernel
def resetForce():
    for i, j, k in fx:
        fx[i, j, k] = 0.0
        fy[i, j, k] = ALPHA * density[i, j, k] - BETA * (temperature[i, j, k] - T_AMBIENT)
        fz[i, j, k] = 0.0


@ti.kernel
def calVorticity():
    for i, j, k in avg_u:
        avg_u[i, j, k] = 0.5 * (u[i, j, k] + u[i + 1, j, k])
    for i, j, k in avg_v:
        avg_v[i, j, k] = 0.5 * (v[i, j, k] + v[i, j + 1, k])
    for i, j, k in avg_w:
        avg_w[i, j, k] = 0.5 * (w[i, j, k] + w[i, j, k + 1])

    for i, j, k in omg_x:
        if 0 < i < Nx - 1 and 0 < j < Ny - 1 and 0 < k < Nz - 1:
            omg_x[i, j, k] = ((avg_w[i, j + 1, k] - avg_w[i, j - 1, k]) - (avg_v[i, j, k + 1] - avg_v[i, j, k - 1])) * 0.5 / VOXEL_SIZE
            omg_y[i, j, k] = ((avg_u[i, j, k + 1] - avg_u[i, j, k - 1]) - (avg_w[i + 1, j, k] - avg_w[i - 1, j, k])) * 0.5 / VOXEL_SIZE
            omg_z[i, j, k] = ((avg_v[i + 1, j, k] - avg_v[i - 1, j, k]) - (avg_u[i, j + 1, k] - avg_u[i, j - 1, k])) * 0.5 / VOXEL_SIZE
        else:
            omg_x[i, j, k] = 0.0
            omg_y[i, j, k] = 0.0
            omg_z[i, j, k] = 0.0

    for i, j, k in fx:
        if 0 < i < Nx - 1 and 0 < j < Ny - 1 and 0 < k < Nz - 1:
            vort_xp = ti.Vector([omg_x[i + 1, j, k], omg_y[i + 1, j, k], omg_z[i + 1, j, k]], dt=ti.f64).norm()
            vort_xm = ti.Vector([omg_x[i - 1, j, k], omg_y[i - 1, j, k], omg_z[i - 1, j, k]], dt=ti.f64).norm()
            grad1 = (vort_xp - vort_xm) * 0.5 / VOXEL_SIZE

            vort_yp = ti.Vector([omg_x[i, j + 1, k], omg_y[i, j + 1, k], omg_z[i, j + 1, k]], dt=ti.f64).norm()
            vort_ym = ti.Vector([omg_x[i, j - 1, k], omg_y[i, j - 1, k], omg_z[i, j - 1, k]], dt=ti.f64).norm()
            grad2 = (vort_yp - vort_ym) * 0.5 / VOXEL_SIZE

            vort_zp = ti.Vector([omg_x[i, j, k + 1], omg_y[i, j, k + 1], omg_z[i, j, k + 1]], dt=ti.f64).norm()
            vort_zm = ti.Vector([omg_x[i, j, k - 1], omg_y[i, j, k - 1], omg_z[i, j, k - 1]], dt=ti.f64).norm()
            grad3 = (vort_zp - vort_zm) * 0.5 / VOXEL_SIZE

            gradVort = ti.Vector([grad1, grad2, grad3], dt=ti.f64)
            norm = gradVort.norm() + 1e-6  # Avoid division by zero
            N_ijk = gradVort / norm

            vorticity = ti.Vector([omg_x[i, j, k], omg_y[i, j, k], omg_z[i, j, k]], dt=ti.f64)
            f = VORT_EPS * VOXEL_SIZE * vorticity.cross(N_ijk)
            vort[i, j, k] = f.norm()
            fx[i, j, k] += f[0]
            fy[i, j, k] += f[1]
            fz[i, j, k] += f[2]


@ti.kernel
def addForce():
    for i, j, k in density:
        if i < Nx - 1:
            u[i + 1, j, k] += DT * 0.5 * (fx[i, j, k] + fx[i + 1, j, k])
        if j < Ny - 1:
            v[i, j + 1, k] += DT * 0.5 * (fy[i, j, k] + fy[i, j + 1, k])
        if k < Nz - 1:
            w[i, j, k + 1] += DT * 0.5 * (fz[i, j, k] + fz[i, j, k + 1])


def build_matrix():
    tripletList = []
    b = np.zeros(Nx * Ny * Nz, dtype=np.float64)
    coeff = VOXEL_SIZE / DT

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                idx = i * Ny * Nz + j * Nz + k
                F = [float(k > 0), float(j > 0), float(i > 0),
                     float(i < Nx - 1), float(j < Ny - 1), float(k < Nz - 1)]
                D = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
                U = [
                    w[i, j, k],
                    v[i, j, k],
                    u[i, j, k],
                    u[i + 1, j, k],
                    v[i, j + 1, k],
                    w[i, j, k + 1]
                ]
                sum_F = sum(F)
                divergence = sum(D[n] * F[n] * U[n] for n in range(6))

                b[idx] = divergence * coeff

                center = -sum_F
                tripletList.append((idx, idx, center))

                if k > 0:
                    idx_km = i * Ny * Nz + j * Nz + (k - 1)
                    tripletList.append((idx, idx_km, F[0]))
                if j > 0:
                    idx_jm = i * Ny * Nz + (j - 1) * Nz + k
                    tripletList.append((idx, idx_jm, F[1]))
                if i > 0:
                    idx_im = (i - 1) * Ny * Nz + j * Nz + k
                    tripletList.append((idx, idx_im, F[2]))
                if i < Nx - 1:
                    idx_ip = (i + 1) * Ny * Nz + j * Nz + k
                    tripletList.append((idx, idx_ip, F[3]))
                if j < Ny - 1:
                    idx_jp = i * Ny * Nz + (j + 1) * Nz + k
                    tripletList.append((idx, idx_jp, F[4]))
                if k < Nz - 1:
                    idx_kp = i * Ny * Nz + j * Nz + (k + 1)
                    tripletList.append((idx, idx_kp, F[5]))

    rows, cols, data = zip(*tripletList)
    A = sp.coo_matrix((data, (rows, cols)), shape=(Nx * Ny * Nz, Nx * Ny * Nz))

    return A, b


def calPressure():
    global A_global, b_global, x_global
    A_global = None
    b_global = None
    x_global = None
    print("*** pressure ***")
    print(pressure[12, 1, 12])

    A_global, b_global = build_matrix()
    print(b_global[10:20])

    x_global, info = cg(A_global, b_global, rtol=tolerance * 1e-8, maxiter=200000)
    if info == 0:
        print("Pressure solver converged successfully.")
    else:
        print(f"Pressure solver did not converge. Info: {info}")

    error = np.linalg.norm(b_global - A_global.dot(x_global))
    if error < tolerance:
        print(f"Converged with error: {error:.2e}")
    else:
        print(f"Not converged. Error: {error:.2e}")

    pressure_res = x_global.reshape((Nx, Ny, Nz))
    print(pressure_res.shape)
    print(pressure_res[12:16, 1:5, 12:16])
    pressure.from_numpy(pressure_res)


@ti.kernel
def applyPressureTerm():
    for i, j, k in ti.ndrange(Nx, Ny, Nz):
        if i < Nx - 1:
            u[i + 1, j, k] -= DT * (pressure[i + 1, j, k] - pressure[i, j, k]) / VOXEL_SIZE
        if j < Ny - 1:
            v[i, j + 1, k] -= DT * (pressure[i, j + 1, k] - pressure[i, j, k]) / VOXEL_SIZE
        if k < Nz - 1:
            w[i, j, k + 1] -= DT * (pressure[i, j, k + 1] - pressure[i, j, k]) / VOXEL_SIZE

    # BUG: AssertionError: copy_from cannot be called in Taichi-scope
    # # Update the old velocity field
    # u0.copy_from(u)
    # v0.copy_from(v)
    # w0.copy_from(w)
    
    # pass
    
def update_velocity_fields():
    u0.copy_from(u)
    v0.copy_from(v)
    w0.copy_from(w)

@ti.func
def get_center(i, j, k):
    half_dx = 0.5 * VOXEL_SIZE
    x = half_dx + i * VOXEL_SIZE
    y = half_dx + j * VOXEL_SIZE
    z = half_dx + k * VOXEL_SIZE
    return ti.Vector([x, y, z], dt=ti.f64)

@ti.func
def get_velocity(pos):
    vel = ti.Vector([get_velocity_x(pos), get_velocity_y(pos), get_velocity_z(pos)], dt=ti.f64)
    return vel

@ti.func
def get_velocity_x(pos):
    return interp_field(u0, pos - ti.Vector([0.0, VOXEL_SIZE * 0.5, VOXEL_SIZE * 0.5], dt=ti.f64))

@ti.func
def get_velocity_y(pos):
    return interp_field(v0, pos - ti.Vector([VOXEL_SIZE * 0.5, 0.0, VOXEL_SIZE * 0.5], dt=ti.f64))

@ti.func
def get_velocity_z(pos):
    return interp_field(w0, pos - ti.Vector([VOXEL_SIZE * 0.5, VOXEL_SIZE * 0.5, 0.0], dt=ti.f64))

@ti.func
def get_density(pos):
    return interp_field(density0, pos - ti.Vector([VOXEL_SIZE * 0.5, VOXEL_SIZE * 0.5, VOXEL_SIZE * 0.5], dt=ti.f64))

@ti.func
def get_temperature(pos):
    return interp_field(temperature0, pos - ti.Vector([VOXEL_SIZE * 0.5, VOXEL_SIZE * 0.5, VOXEL_SIZE * 0.5], dt=ti.f64))

@ti.func
def interp_field(field, pos):
    if ti.static(INTERPOLATION_METHOD == InterpolationMethod.LINEAR):
        return linear_interpolation(field, pos)
    elif ti.static(INTERPOLATION_METHOD == InterpolationMethod.MONOTONIC_CUBIC):
        return monotonic_cubic_interpolation(field, pos)
    
@ti.func
def linear_interpolation(field, pos):
    x = pos[0] / VOXEL_SIZE - 0.5
    y = pos[1] / VOXEL_SIZE - 0.5
    z = pos[2] / VOXEL_SIZE - 0.5
    i = ti.cast(ti.floor(x), ti.i32)
    j = ti.cast(ti.floor(y), ti.i32)
    k = ti.cast(ti.floor(z), ti.i32)
    i = ti.max(0, ti.min(i, field.shape[0] - 2))
    j = ti.max(0, ti.min(j, field.shape[1] - 2))
    k = ti.max(0, ti.min(k, field.shape[2] - 2))
    ix = x - i
    iy = y - j
    iz = z - k
    res = (field[i, j, k] * (1 - ix) * (1 - iy) * (1 - iz) +
            field[i + 1, j, k] * ix * (1 - iy) * (1 - iz) +
            field[i, j + 1, k] * (1 - ix) * iy * (1 - iz) +
            field[i, j, k + 1] * (1 - ix) * (1 - iy) * iz +
            field[i + 1, j + 1, k] * ix * iy * (1 - iz) +
            field[i + 1, j, k + 1] * ix * (1 - iy) * iz +
            field[i, j + 1, k + 1] * (1 - ix) * iy * iz +
            field[i + 1, j + 1, k + 1] * ix * iy * iz)
    return res

@ti.func
def monotonic_cubic_interpolation(field, pos):
    x = pos[0] / VOXEL_SIZE - 0.5
    y = pos[1] / VOXEL_SIZE - 0.5
    z = pos[2] / VOXEL_SIZE - 0.5
    i = ti.cast(ti.floor(x), ti.i32)
    j = ti.cast(ti.floor(y), ti.i32)
    k = ti.cast(ti.floor(z), ti.i32)
    i = ti.max(1, ti.min(i, field.shape[0] - 3))
    j = ti.max(1, ti.min(j, field.shape[1] - 3))
    k = ti.max(1, ti.min(k, field.shape[2] - 3))
    fx = x - i
    fy = y - j
    fz = z - k

    arr_z = ti.Vector([0.0, 0.0, 0.0, 0.0], dt=ti.f64)
    for z in ti.static(range(4)):
        arr_x = ti.Vector([0.0, 0.0, 0.0, 0.0], dt=ti.f64)
        for x in ti.static(range(4)):
            arr_y = ti.Vector([field[i + x - 1, j - 1, k + z - 1],
                               field[i + x - 1, j, k + z - 1],
                               field[i + x - 1, j + 1, k + z - 1],
                               field[i + x - 1, j + 2, k + z - 1]], dt=ti.f64)
            arr_x[x] = axis_monotonic_cubic_interpolation(arr_y, fy)
        arr_z[z] = axis_monotonic_cubic_interpolation(arr_x, fx)
    return axis_monotonic_cubic_interpolation(arr_z, fz)

@ti.func
def c_sign(x):
    return -1.0 if x < 0 else (1.0 if x > 0 else 0.0)

@ti.func
def axis_monotonic_cubic_interpolation(f, t):
    delta = f[2] - f[1]
    d0 = 0.5 * (f[2] - f[0])
    d1 = 0.5 * (f[3] - f[1])

    d0 = c_sign(delta) * ti.abs(d0)
    d1 = c_sign(delta) * ti.abs(d1)

    a0 = f[1]
    a1 = d0
    a2 = 3 * delta - 2 * d0 - d1
    a3 = d0 + d1 - 2 * delta
    return a3 * t * t * t + a2 * t * t + a1 * t + a0

@ti.kernel
def advectVelocity():
    # print("*** advectVelocity ***")
    if ADVECTION_METHOD == AdvectionMethod.SEMI_LAGRANGE:
        for i, j, k in ti.ndrange(Nx + 1, Ny, Nz): # u
            pos_u = get_center(i, j, k) - ti.Vector([VOXEL_SIZE * 0.5, 0.0, 0.0], dt=ti.f64)
            vel_u = get_velocity(pos_u)
            pos_u -= DT * vel_u
            u[i, j, k] = get_velocity_x(pos_u)
        for i, j, k in ti.ndrange(Nx, Ny + 1, Nz):
            pos_v = get_center(i, j, k) - ti.Vector([0.0, VOXEL_SIZE * 0.5, 0.0], dt=ti.f64)
            vel_v = get_velocity(pos_v)
            pos_v -= DT * vel_v
            v[i, j, k] = get_velocity_y(pos_v)
        for i, j, k in ti.ndrange(Nx, Ny, Nz + 1):
            pos_w = get_center(i, j, k) - ti.Vector([0.0, 0.0, VOXEL_SIZE * 0.5], dt=ti.f64)
            vel_w = get_velocity(pos_w)
            pos_w -= DT * vel_w
            w[i, j, k] = get_velocity_z(pos_w)
    elif ADVECTION_METHOD == AdvectionMethod.MAC_CORMACK:
        # print("*** MAC_CORMACK ***")
        for i, j, k in u:
            # print(f"u: {i}, {j}, {k}")
            u_n = u0[i, j, k]
            pos_u = get_center(i, j, k) - ti.Vector([0.5 * VOXEL_SIZE, 0, 0], dt=ti.f64)
            vel_u = get_velocity(pos_u)
            # forward advection
            pos_u -= DT * vel_u
            u_np1_hat = get_velocity_x(pos_u)
            # backward advection
            pos_u += DT * get_velocity(pos_u)
            u_n_hat = get_velocity_x(pos_u)
            u[i, j, k] = u_np1_hat + 0.5 * (u_n - u_n_hat)

        for i, j, k in v:
            v_n = v0[i, j, k]
            pos_v = get_center(i, j, k) - ti.Vector([0, 0.5 * VOXEL_SIZE, 0], dt=ti.f64)
            vel_v = get_velocity(pos_v)
            # forward advection
            pos_v -= DT * vel_v
            v_np1_hat = get_velocity_y(pos_v)
            # backward advection
            pos_v += DT * get_velocity(pos_v)
            v_n_hat = get_velocity_y(pos_v)
            v[i, j, k] = v_np1_hat + 0.5 * (v_n - v_n_hat)

        for i, j, k in w:
            w_n = w0[i, j, k]
            pos_w = get_center(i, j, k) - ti.Vector([0, 0, 0.5 * VOXEL_SIZE], dt=ti.f64)
            vel_w = get_velocity(pos_w)
            # forward advection
            pos_w -= DT * vel_w
            w_np1_hat = get_velocity_z(pos_w)
            # backward advection
            pos_w += DT * get_velocity(pos_w)
            w_n_hat = get_velocity_z(pos_w)
            w[i, j, k] = w_np1_hat + 0.5 * (w_n - w_n_hat)

def update_scalar_fields():
    density0.copy_from(density)
    temperature0.copy_from(temperature)

@ti.kernel
def advectScalar():
    # density0.copy_from(density)
    # temperature0.copy_from(temperature)

    if ADVECTION_METHOD == AdvectionMethod.SEMI_LAGRANGE:
        for i, j, k in ti.ndrange(Nx, Ny, Nz):
            pos_cell = get_center(i, j, k)
            vel_cell = get_velocity(pos_cell)
            pos_cell -= DT * vel_cell
            density[i, j, k] = get_density(pos_cell)
            temperature[i, j, k] = get_temperature(pos_cell)
    elif ADVECTION_METHOD == AdvectionMethod.MAC_CORMACK:
        for i, j, k in density:
            d_n = density[i, j, k]
            t_n = temperature[i, j, k]
            pos_cell = get_center(i, j, k)
            vel_cell = get_velocity(pos_cell)
            # forward advection
            pos_cell -= DT * vel_cell
            d_np1_hat = get_density(pos_cell)
            t_np1_hat = get_temperature(pos_cell)
            # backward advection
            pos_cell += DT * get_velocity(pos_cell)
            d_n_hat = get_density(pos_cell)
            t_n_hat = get_temperature(pos_cell)
            density[i, j, k] = d_np1_hat + 0.5 * (d_n - d_n_hat)
            temperature[i, j, k] = t_np1_hat + 0.5 * (t_n - t_n_hat)
    

def update():
    
    # print("*** 0 ***")
    resetForce()
    # print("*** 1 ***")
    calVorticity()
    # print("*** 2 ***")
    addForce()
    # print("*** 3 ***")
    calPressure()
    # print("*** 4 ***")
    applyPressureTerm()
    # print("*** 5 ***")
    update_velocity_fields()
    # print("*** 6 ***")
    advectVelocity()
    # print("*** 7 ***")
    update_scalar_fields()
    # print("*** 8 ***")
    advectScalar()
     
    if time[None] < EMIT_DURATION:
        addSource()
        setEmitterVelocity()
    time[None] += DT


def save_density_to_vdb(frame):
    density_np = density.to_numpy()
    grid = vdb.FloatGrid()
    accessor = grid.getAccessor()
    grid.name = "frame_" + str(frame)

    for i in range(density_np.shape[0]):
        for j in range(density_np.shape[1]):
            for k in range(density_np.shape[2]):
                accessor.setValueOn((i, j, k), float(density_np[i, j, k]))

    vdb.write(f"frames/frame_{frame:04d}.vdb", grids=[grid])


def main():
    os.makedirs("frames", exist_ok=True)
    initialize()
    addSource()
    setEmitterVelocity()
    frame = 0
    while time[None] < FINISH_TIME:
        update()
        save_density_to_vdb(frame)
        frame += 1
        print(f"Frame {frame}, Time {time[None]:.2f}")

main()
