import os
import torch
import pickle

from config import Config
from models.experiments import SimulationExperiment
from simulation.model import Model
from simulation.initial_distributions import InitialDistributions
from simulation.model_callback import ModelCallback
from simulation.transition_functions import TransitionFunctions
from simulation.policies import Policies
from simulation.impl import *
from utils.datetime import now_str


class Simulation:
    def __init__(
        self,
        experiment: SimulationExperiment,
        model_callback: ModelCallback = None,
    ):
        self.experiment = experiment
        self.model_callback = model_callback

        self.model_context = self.experiment.model_context

        self.initial_distributions = InitialDistributions(self.model_context)

        self.transition_functions = TransitionFunctions(
            O=OTransitionFunctionDefault(),
            To=ToTransitionFunctionDefault(),
            Q=QTransitionFunctionDefault(self.model_context),
            Tq=TqTransitionFunctionDefault(self.model_context),
            A=ATransitionFunctionDefault(
                model_context=self.model_context,
                resource_models=self.experiment.available_resource_models,
                server_means=self.experiment.server_available_resource_means,
                server_stds=self.experiment.server_available_resource_stds,
                server_maxs=self.experiment.server_available_resource_maxs,
                client_means=self.experiment.client_available_resource_means,
                client_stds=self.experiment.client_available_resource_stds,
                client_maxs=self.experiment.client_available_resource_maxs,
            ),
            C=CTransitionFunctionDefault(
                model_context=self.model_context,
                x_resource_models=self.experiment.x_resource_consumption_models,
                y_downloading_resource_models=self.experiment.y_download_resource_consumption_models,
                y_executing_resource_models=self.experiment.y_execution_resource_consumption_models,
                y_uploading_resource_models=self.experiment.y_upload_resource_consumption_models,
            ),
            Xhat=XhatTransitionFunctionDefault(),
            Tx=TxTransitionFunctionDefault(self.model_context),
            X=XTransitionFunctionDefault(),
            Yihat=YihatTransitionFunctionDefault(),
            Yshat=YshatTransitionFunctionDefault(),
            Ty=TyTransitionFunctionDefault(
                model_context=self.model_context,
                task_completion_models=self.experiment.y_task_completion_models,
            ),
            Yi=YiTransitionFunctionDefault(),
            Ys=YsTransitionFunctionDefault(),
            W=WTransitionFunctionDefault(),
        )

        self.policies = Policies(
            server=self.experiment.server_policy,
            client=self.experiment.client_policy,
        )

        self.trajectory = self.experiment.trajectory

        self.model = Model(
            model_context=self.model_context,
            initial_distributions=self.initial_distributions,
            transition_functions=self.transition_functions,
            policies=self.policies,
            trajectory=self.trajectory,
            model_callback=self.model_callback,
        )

    def simulate(self):
        with torch.inference_mode():
            self.model.run()

    def save_trajectory(self):
        config = Config()
        handle = self.experiment.experiment_handle

        os.makedirs(config.temp_dir, exist_ok=True)
        file_path = os.path.join(
            config.temp_dir, f"trajectory_exp_{handle}_{now_str()}.pkl")

        with open(file_path, "wb") as f:
            pickle.dump(self.trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)

        return file_path
