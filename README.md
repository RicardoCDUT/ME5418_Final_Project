
# PyBox2D Environment Setup

This project uses the `Box2D` physics engine with `Python 3.6` to simulate 2D environments. Below are instructions to set up the development environment using Conda.

## Prerequisites

- **Conda**: Ensure you have `Conda` installed. You can download it from the official [Anaconda](https://www.anaconda.com/) website or use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for a lightweight version.

## Setting Up the Conda Environment

You can use the provided `environment.yml` file to create the environment:

1. Save the `environment.yml` file.
2. Run the following command to create the environment:

   ```bash
   conda env create -f environment.yml
   ```

Alternatively, if you don't have the `environment.yml` file, you can manually create the environment with the following command:

```bash
conda create -n pybox2d -c conda-forge python=3.6 pybox2d
```

After creating the environment, activate it:

```bash
conda activate pybox2d
```
## Project Structure
- main.py: The entry point for the simulation, where the Gym environment (envGym) is instantiated and the rendering loop is started.
- unit_test: A testing file that includes render(), rest(), step(), action space(), and state space().
- show_main: Omit the training section and directly display the final result.
- agent.py: Contains the Q-agent class responsible for interacting with the environment.
- memory_buffer.py: Implements a memory buffer (or replay buffer) to store experience tuples (state, action, reward, next state, end_flag) collected during training.
- environment.py: Sets up the simulation environment using PyBox2D and defines the physics and dynamics for objects in the environment. 
- qNetwork.py: Defines the neural network model for the Q-learning algorithm, responsible for approximating the Q-values.
## Running the Code

Once the environment is activated, you can run the simulation using:

```bash
python show_main.py
```
If you want to learn the final result, please run the `show_main.py`, where the agent will directly load agent.pth from the root directory.

```bash
python main.py
```

Make sure that `main.py` is the entry point for your simulation, where the Gym environment (`envGym`) is instantiated and the rendering loop is started.

## Unit test

```bash
python unit_test.py
```
Simple unit testing for final report.

## External Dependencies

- **PyBox2D**: The environment uses the `PyBox2D` package for physics simulation.
- **Gym (if applicable)**: If you're using OpenAI's Gym framework, ensure that it is installed in the environment. You can add it by running:

  ```bash
  conda install -c conda-forge gym
  ```

## Notes

- If you encounter issues, check if you are using the correct Python version (3.6) and that all required libraries (like `pybox2d`) are installed.
- The version is CUDA, Please notice to switching your PyTorch mode.

## Troubleshooting

If you face any problems with missing dependencies, consider adding them manually to the environment using `conda install` or updating the `environment.yml` file.
