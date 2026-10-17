import torch
import argparse

from models.experiments import SimulationExperiment
from simulation.simulation import Simulation
from simulation.impl import *


def run(args):
    experiment = SimulationExperiment(args.experiment)

    torch.set_default_dtype(experiment.model_context.float_type)
    torch.set_default_device(experiment.model_context.device)

    simulation = Simulation(experiment)
    simulation.simulate()
    file_path = simulation.save_trajectory()

    print(f"Saved trajectory to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str,
                        required=True, help="Experiment handle")

    args = parser.parse_args()
    run(args)
