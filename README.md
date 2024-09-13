# NMR Simulation Tool

This repository contains a Rust-based tool for simulating Nuclear Magnetic Resonance (NMR) signals as recorded using nitrogen vacancy centers in diamond.

## Features

- **Mesh-based sample geometry**: Specify the sample volume and geometry in STL format.
- **Parallel Processing**: Utilizes data parallelism to speed up the simulation.
- **Customizable Simulation Parameters**: Allows detailed configuration of the simulation via a YAML input file.

## Requirements

- Rust (1.56 or newer)
- Cargo (Rust's package manager)
- HDF5 (for output)

## Setup

### 1. Install Rust

If you don't have Rust installed, you can install it using `rustup`:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

After installation, ensure that Cargo and Rust are available in your path:

bash

rustc --version
cargo --version

2. Clone the Repository

Clone the repository to your local machine:

bash

git clone https://github.com/yourusername/nmr-simulation-tool.git
cd nmr-simulation-tool

3. Build the Project

Use Cargo to build the project:

bash

cargo build --release

This will generate an optimized binary in the target/release directory.
Usage
Input YAML File

The simulation requires an input YAML file that defines the simulation parameters. Below is an example of how the YAML file should look:

yaml

m1x: 0.0
m1y: 0.81654
m1z: 0.57728
m2x: 0.0
m2y: -0.57728
m2z: 0.81654
nv_depth: 10.0
proton_count: 36000000
output_file: out_cli10.h5
stl_file: ChipNew.stl
resolution_x: 100
resolution_y: 100
diffusion_coefficient: 2299.0e-6
frequency: 3570000.0
number_time_steps: 1500
timestep: 1.8674e-9
scale_factor: 50
parallelization_level: 10

Running the Simulation

Once you have your YAML configuration file prepared (let's say it's named config.yaml), you can run the simulation as follows:

bash

./target/release/nmr_simulation_tool config.yaml

Output

The output of the simulation is an HDF5 file specified in the output_file parameter of the YAML configuration. This file contains the results of the NMR simulation.
Example

Assuming your YAML file is named config.yaml and located in the project root, and your STL file is ChipNew.stl, you would run the simulation like this:

bash

./target/release/nmr_simulation_tool config.yaml

The output will be saved as out_cli10.h5 (or whatever filename you provided in the YAML file).
Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request. We welcome all contributions!
License

This project is licensed under the MIT License. See the LICENSE file for more details.
