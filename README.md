# ACG-Project

## Code Structure

```
ACG-Project
|   src
|   │   material 
|   |   |   |   fluid
|   |   |   |   container
|   |   |   |   other materials
|   │   |   # Material for simulation
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

### Fluid Simulation

https://github.com/user-attachments/assets/ca0cbeda-95f9-48c9-b504-afa252acb9d9

### Large Scale Fluid Simulation (DFSPH)

https://github.com/user-attachments/assets/5f6af6e3-34b0-4b58-8abf-ba0c50f65d2e

### Rigid-Fluid Interaction (WCSPH)

https://github.com/user-attachments/assets/a4f351b3-da79-40de-9e15-1973ab6f4ab7

### Cloth Simulation

https://github.com/user-attachments/assets/74b2ef8f-fb27-43f5-8ba6-3b241366163b

### Cloth-Rigid Interaction

https://github.com/user-attachments/assets/4028f1b6-e5ae-431b-982a-d30c5923775f

https://github.com/user-attachments/assets/f2b8f1a9-a5af-4f2b-a8ee-b79fccb92ddb


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
