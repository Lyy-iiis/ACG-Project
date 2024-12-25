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
|   doc # Documentation
|   main.py
|   UPDATE_LOG.md
|   README.md
```

## Demo

### Rigid-Fluid Interaction (WCSPH and DFSPH)

Fixed Rigid body

https://github.com/user-attachments/assets/0394d752-4cb5-4227-a11e-d18ef5b025be

https://github.com/user-attachments/assets/a49996d7-b11d-444c-b0a9-9970bcfc52a9

Movable rigid body

https://github.com/user-attachments/assets/c3759e06-526b-4526-b1f9-29ae901550a9

### Cloth-Rigid Interaction

Fixed rigid body

https://github.com/user-attachments/assets/f2b8f1a9-a5af-4f2b-a8ee-b79fccb92ddb

Movable rigid body

https://github.com/user-attachments/assets/aca1b0f9-4a01-4ecc-82eb-214dd6e589b6

https://github.com/user-attachments/assets/2d6644f5-2611-4fcc-97ae-ef2f20a1c7fd

### Realtime interactive rendering

https://github.com/user-attachments/assets/ecb0b677-2147-4369-bfa3-cdd3463f25ef

For more videos, please refer to the [doc/video.md](doc/video.md).

### Smoke Simulation

https://github.com/user-attachments/assets/e734c73b-95f0-43e7-82ce-6c4bada9730b

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
### TBD

1. Code for PCISPH waitting for update

2. Speed up for DFSPH

3. Modularize code
