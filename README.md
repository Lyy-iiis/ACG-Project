# ACG-Project

## Code Structure

```
ACG-Project
|   src
|   │   material 
|   |   |   # Material for the project
|   │   render 
|   |   |   # Render the scene
|   │   visualize
|   |   |   # File for visualization
|   assets # obj files
|   main.py
|   UPDATE_LOG.md
|   README.md
```

## Demo

<figure align="center">
  <video src="./examples/fluid_small.mp4" alt="Fluid" width="96%" controls>
  </video>
  <figcaption>Fluid Simulation</figcaption>
</figure>
<figure align="center">
  <video src="./examples/fluid_large.mp4" alt="Fluid" width="96%" controls>
  </video>
  <figcaption>Large Scale Fluid Simulation</figcaption>
</figure>

https://github.com/Lyy-iiis/ACG-Project/blob/main/examples/fluid_large.mp4

## Installation

1. Clone the repository

```bash
sudo apt-get install git-lfs
git lfs install
git clone git@github.com:Lyy-iiis/ACG-Project.git
git lfs pull
```

2. Install the required packages

```bash
conda env create -f environment.yml
```

3. Install required packages for blender
  
```bash
sudo apt-get install libxrender1
sudo apt-get install libxi6
sudo apt-get install libxkbcommon0
sudo apt-get install libsm6
```

4. Install rust and splashsurf

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install splashsurf
```

5. Activate the environment

```bash
conda activate acg
```

6. Run the main.py

```bash
python main.py
```
