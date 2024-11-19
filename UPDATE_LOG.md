# Update Log

## 24.11.19 16:00 TXC

- Finish basic cloth-rigid coupling, change the viewpoint

## 24.11.19 11:45 TXC

- Modify the cloth rendering architecture to be compatible with the rigid body and fluid architectures
- Finish implementing double-sided rendering of cloth
- Add basic self-collision

## 24.11.18 8:30 LYY

- Finish DFSPH rigid coupling, now the fluid can interact with rigid body correctly.

## 24.11.17 19:30 LYY

- Finish complete DFSPH, however the rendering is a bit slower.

## 24.11.15 8:00 LYY

- After two days of debugging, finally implement DFSPH correctly.

## 24.11.13 9:30 LYY

- Refactor code structure, moved basic content to base class.
- Init DFSPH

## 24.10.31 21:00 LYY

- Increase capability of Container.
- Fix all warning in compilation.

## 24.10.29 16:30 LYY

- Take large effort to make render scene more beautiful, now the rendering effect is more realistic.

## 24.10.28 21:00 LYY

- Implement fluid-rigid coupling.

## 24.10.28 8:30 LYY

- Move fluid neighbor update to container class, now the fluid simulation is more modular.
- Initialize fluid and rigid coupling.

## 24.10.26 13:00 LYY

- Speed up initialization of fluid particles.
- Try longer term fluid simulation.

## 24.10.25 22:30 LYY

- Implement position hash grid, now the fluid can be simulated more efficiently. From $O(n^2)$ to $O(n)$.

## 24.10.25 16:00 LYY

- Finish surface tension, now the fluid can be simulated more realistically.
- Render long term fluid simulation result correctly.
- More beautiful rendering effect.
- TODO: fluid render speed up.

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
