# ACG-Project

## Code Structure
```
ACG-Project
|   src
|   │   material 
|   |   |   # Material for the project
|   │   render 
|   |   |   # Render the scene
|   │   surface.py
|   |   |   # Surface reconstruction
|   assets # obj files
|   main.py
|   UPDATE_LOG.md
|   README.md
```

## Installation

1. Clone the repository
```bash
git clone git@github.com:Lyy-iiis/ACG-Project.git
```

2. Install the required packages
```bash
conda env create -f environment.yml
```

3. Install rust and splashsurf
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install splashsurf
```

4. Activate the environment
```bash
conda activate acg
```

5. Run the main.py
```bash
python main.py
```