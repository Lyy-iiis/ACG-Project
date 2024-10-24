# Update Log

## 24.10.24 20:00 LYY

- Finish container boundary condition, now fluid move correctly.
- Clean up the code, more modular and readable.

## 24.10.24 13:00 LYY

- Finish WCSPH, including pressure force, viscosity force, gravity force, now the fluid can be simulated correctly.
- TODO: container boundary condition, surface tension.

## 24.10.24 10:00 LYY

- Prepare the basic feature for SPH.
- Change the sampling method from random to regular grid.

## 24.10.23 16:30 LYY

- Finish surface reconstruction from fluid particles.
- Implement basic container.

## 24.10.20 10:30 TXC

- Modify cloth inplementation.

## 24.10.12 16:30 TXC

- Implement basic cloth class, using Mass-Spring model.
- Implement basic collision detection.
- TODO: here are some bugs in the cloth rendering codes.

## 24.10.09 14:00 LYY

- Implement basic fluid class, using Lagrangian method to simulate fluid.
- Implement basic container class, which can contain fluid.
- TODO: recover mesh from fluid particles for rendering.

## 24.10.02 22:45 LYY

- Implement basic geometric shapes
- Now support multi-object rendering

## 24.10.02 17:30 LYY

- Implement the inertia tensor, now angular velocity is computed correctly.
- Fix the bug in video recording
- Clean up the code, more modular and readable.

## 24.10.02 15:00 LYY

- Finish basic feature of rigid body simulation, such as applying force, update position and velocity.
- Change main data structure from numpy to taichi vector.

## 24.9.30 20:00 LYY

- Init rendering engine and rigid body simulation engine.
