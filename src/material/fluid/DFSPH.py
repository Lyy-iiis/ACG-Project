import taichi as ti
import numpy as np
from ..utils import *
from src.material.fluid.basefluid import BaseFluid

@ti.data_oriented
class DFSPH(BaseFluid):
    def __init__(self, mesh: trimesh.Trimesh,
                 position=np.array([0.0, 0.0, 0.0]), 
                 gravity=np.array([0.0, -9.8, 0.0]), 
                 viscosity=10, rest_density=1000,
                 time_step=0.0005, fps=60):
        super().__init__(mesh, position, gravity, viscosity, rest_density, time_step, fps)
    
        self.m_max_iterations_v = 1000
        self.m_max_iterations = 1000

        self.m_eps = 1e-5

        self.max_error_V = 0.001
        self.max_error = 0.0001