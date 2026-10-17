import torch
import pyro.distributions as dist
from math import prod
from collections import defaultdict

from simulation.types import *
from simulation.model_context import ModelContext
from simulation.transition_functions import CTransitionFunction
from utils.resource_consumption_model import ResourceConsumptionModel
from utils.vector_arima_simulator import VectorArimaSimulator, SimulatorConfig
from utils.unique import *


class CTransitionFunctionDefault(CTransitionFunction):

    def __init__(
        self,
        model_context: ModelContext,
        x_resource_models: list[ResourceConsumptionModel],
        y_downloading_resource_models: list[ResourceConsumptionModel],
        y_executing_resource_models: list[ResourceConsumptionModel],
        y_uploading_resource_models: list[ResourceConsumptionModel],
    ):
        assert len(x_resource_models) == len(Resource.X_resources())
        assert len(y_downloading_resource_models) == len(
            Resource.Y_resources(YState.DOWNLOADING))
        assert len(y_executing_resource_models) == len(
            Resource.Y_resources(YState.EXECUTING))
        assert len(y_uploading_resource_models) == len(
            Resource.Y_resources(YState.UPLOADING))

        self._model_context = model_context
        self._x_models = x_resource_models
        self._y_downloading_models = y_downloading_resource_models
        self._y_executing_models = y_executing_resource_models
        self._y_uploading_models = y_uploading_resource_models

        self._x_simulators = self._get_x_simulators(x_resource_models)
        self._y_downloading_simulators = self._get_y_simulators(
            y_downloading_resource_models, state=YState.DOWNLOADING)
        self._y_executing_simulators = self._get_y_simulators(
            y_executing_resource_models, state=YState.EXECUTING)
        self._y_uploading_simulators = self._get_y_simulators(
            y_uploading_resource_models, state=YState.UPLOADING)

        self._x_configs = self._get_x_configs(x_resource_models)
        self._y_downloading_configs = self._get_y_configs(
            y_downloading_resource_models, state=YState.DOWNLOADING)
        self._y_executing_configs = self._get_y_configs(
            y_executing_resource_models, state=YState.EXECUTING)
        self._y_uploading_configs = self._get_y_configs(
            y_uploading_resource_models, state=YState.UPLOADING)

        self._storage_cumsum = torch.zeros(
            (*model_context.partition_sizes.shape, model_context.n_nodes))

    def __call__(self, Cx_prev, Cy_prev, Xhat_curr, Yihat_curr, Yshat_curr, Tx_prev, Ty_prev, A_curr):
        bandwidth_in_x_idx = Resource.X_resource_idx(Resource.BANDWIDTH_IN)
        bandwidth_in_y_downloading_idx = Resource.Y_resource_idx(
            Resource.BANDWIDTH_IN, state=YState.DOWNLOADING)

        # Bandwidth IN is shared between Cx and Cy, so we need to scale both based on total consumption first.

        bandwidth_in_x_model = self._x_models[bandwidth_in_x_idx]
        bandwidth_in_x_config = self._x_configs[bandwidth_in_x_idx]
        bandwidth_in_x_simulator = self._x_simulators[bandwidth_in_x_idx]

        bandwidth_in_y_downloading_model = self._y_downloading_models[
            bandwidth_in_y_downloading_idx]
        bandwidth_in_y_downloading_config = self._y_downloading_configs[
            bandwidth_in_y_downloading_idx]
        bandwidth_in_y_downloading_simulator = self._y_downloading_simulators[
            bandwidth_in_y_downloading_idx]

        # The server is never downloading or uploading, so it won't contribute to these numbers' calculation

        n_downloading = (Xhat_curr == XState.DOWNLOADING).any(dim=(0, 1)) | (
            Yshat_curr == YState.DOWNLOADING).any(dim=(0, 1, 3))
        # Make sure we count only online nodes
        n_downloading &= A_curr[Resource.BANDWIDTH_IN] > 0
        n_downloading = n_downloading.sum().item()

        n_uploading = (Yshat_curr == YState.UPLOADING).any(dim=(0, 1, 3))
        # Make sure we count only online nodes
        n_uploading &= A_curr[Resource.BANDWIDTH_OUT] > 0
        n_uploading = n_uploading.sum().item()

        # Clip client bandwidths in/out based on server bandwidths out/in.
        A_temp = A_curr.clone()
        if n_downloading > 0:
            A_temp[Resource.BANDWIDTH_IN, 1:] = torch.clip(
                A_temp[Resource.BANDWIDTH_IN, 1:], max=A_temp[Resource.BANDWIDTH_OUT, 0] / n_downloading)
        if n_uploading > 0:
            A_temp[Resource.BANDWIDTH_OUT, 1:] = torch.clip(
                A_temp[Resource.BANDWIDTH_OUT, 1:], max=A_temp[Resource.BANDWIDTH_IN, 0] / n_uploading)

        bandwidth_in_x_config.update(
            max=A_temp[Resource.BANDWIDTH_IN],
        )
        bandwidth_in_y_downloading_config.update(
            max=A_temp[Resource.BANDWIDTH_IN],
        )

        Cx_bandwidth_in, _ = self._get_x_consumptions(
            models=[bandwidth_in_x_model],
            configs=[bandwidth_in_x_config],
            simulators=[bandwidth_in_x_simulator],
            Xhat_curr=Xhat_curr,
            A_curr=A_temp,
        )
        Cx_bandwidth_in = Cx_bandwidth_in[0]

        Cy_bandwidth_in, _, _, _, _, _ = self._get_y_consumptions(
            downloading_models=[bandwidth_in_y_downloading_model],
            executing_models=[],
            uploading_models=[],
            downloading_configs=[bandwidth_in_y_downloading_config],
            executing_configs=[],
            uploading_configs=[],
            downloading_simulators=[bandwidth_in_y_downloading_simulator],
            executing_simulators=[],
            uploading_simulators=[],
            Yshat_curr=Yshat_curr,
            Ty_prev=Ty_prev,
            A_curr=A_temp,
        )
        Cy_bandwidth_in = Cy_bandwidth_in[0]

        C_bandwidth_in = Cx_bandwidth_in + Cy_bandwidth_in
        bandwidth_in_scale_factor = torch.nan_to_num(torch.clip(
            A_temp[Resource.BANDWIDTH_IN] / C_bandwidth_in, max=1.0), nan=0.0)

        Cx_bandwidth_in *= bandwidth_in_scale_factor
        Cy_bandwidth_in *= bandwidth_in_scale_factor
        bandwidth_in_scale_factor = None  # free memory

        A_temp[Resource.BANDWIDTH_IN] = Cx_bandwidth_in
        bandwidth_in_x_config.update(
            max=Cx_bandwidth_in,
        )

        models = self._x_models.copy()
        configs = self._x_configs.copy()
        simulators = self._x_simulators.copy()
        models.remove(bandwidth_in_x_model)
        configs.remove(bandwidth_in_x_config)
        simulators.remove(bandwidth_in_x_simulator)

        Cx_consumptions, n_x_active_tasks = self._get_x_consumptions(
            models=models,
            configs=configs,
            simulators=simulators,
            Xhat_curr=Xhat_curr,
            A_curr=A_temp,
        )
        Cx_consumptions = torch.cat([
            Cx_consumptions[:bandwidth_in_x_idx],
            Cx_bandwidth_in.unsqueeze(0),
            Cx_consumptions[bandwidth_in_x_idx:],
        ])
        Cx_bandwidth_in = None  # free memory

        A_temp[Resource.BANDWIDTH_IN] = Cy_bandwidth_in
        bandwidth_in_y_downloading_config.update(
            max=Cy_bandwidth_in,
        )
        bandwidth_out_y_idx = Resource.Y_resource_idx(
            Resource.BANDWIDTH_OUT, state=YState.UPLOADING)
        bandwidth_out_y_downloading_config = self._y_uploading_configs[bandwidth_out_y_idx]
        bandwidth_out_y_downloading_config.update(
            max=A_temp[Resource.BANDWIDTH_OUT],
        )

        y_downloading_models = self._y_downloading_models.copy()
        y_downloading_configs = self._y_downloading_configs.copy()
        y_downloading_simulators = self._y_downloading_simulators.copy()
        y_downloading_models.remove(bandwidth_in_y_downloading_model)
        y_downloading_configs.remove(bandwidth_in_y_downloading_config)
        y_downloading_simulators.remove(bandwidth_in_y_downloading_simulator)

        Cy_downloading_consumptions, Cy_executing_consumptions, Cy_uploading_consumptions, n_y_downloading_tasks, n_y_executing_counts, n_y_uploading_tasks = self._get_y_consumptions(
            downloading_models=y_downloading_models,
            executing_models=self._y_executing_models,
            uploading_models=self._y_uploading_models,
            downloading_configs=y_downloading_configs,
            executing_configs=self._y_executing_configs,
            uploading_configs=self._y_uploading_configs,
            downloading_simulators=y_downloading_simulators,
            executing_simulators=self._y_executing_simulators,
            uploading_simulators=self._y_uploading_simulators,
            Yshat_curr=Yshat_curr,
            Ty_prev=Ty_prev,
            A_curr=A_temp,
        )
        Cy_downloading_consumptions = torch.cat([
            Cy_downloading_consumptions[:bandwidth_in_y_downloading_idx],
            Cy_bandwidth_in.unsqueeze(0),
            Cy_downloading_consumptions[bandwidth_in_y_downloading_idx:],
        ])
        Cy_bandwidth_in = None  # free memory

        # Calculate executing consumptions before, because they are already across partitions, tasks and nodes
        for i in range(Cy_executing_consumptions.shape[0]):
            Cy_executing_consumptions[i] /= torch.clip(
                n_y_executing_counts, min=1.0)

        # Calculate overflow of CPU resources

        total_cpu_consumptions = torch.zeros(
            A_curr.shape[1:], dtype=A_curr.dtype)

        cpu_idx = Resource.X_resource_idx(Resource.CPU)
        total_cpu_consumptions += Cx_consumptions[cpu_idx]

        cpu_y_idx = Resource.Y_resource_idx(
            Resource.CPU, state=YState.DOWNLOADING)
        total_cpu_consumptions += Cy_downloading_consumptions[cpu_y_idx]

        cpu_y_idx = Resource.Y_resource_idx(
            Resource.CPU, state=YState.EXECUTING)
        # Make sure we don't need a mask here, i.e., Cy_executing_consumptions only has non-zero values for executing tasks
        total_cpu_consumptions += Cy_executing_consumptions[cpu_y_idx].reshape(
            self._get_y_dimensions(YState.EXECUTING)).sum(dim=(0, 1, 3))

        cpu_y_idx = Resource.Y_resource_idx(
            Resource.CPU, state=YState.UPLOADING)
        total_cpu_consumptions += Cy_uploading_consumptions[cpu_y_idx]

        # Here we could use online mask to reduce computations

        scale_factor = torch.clip(
            A_curr[Resource.CPU] / total_cpu_consumptions, max=1.0)
        scale_factor = torch.nan_to_num(scale_factor, nan=0.0)
        # overflow_mask = scale_factor < 1.0

        scaled_resource_indices = [
            Resource.X_resource_idx(r)
            for r in {Resource.CPU, Resource.BANDWIDTH_IN, Resource.STORAGE}
        ]
        Cx_consumptions[scaled_resource_indices] *= scale_factor[None]

        scaled_resource_indices = [
            Resource.Y_resource_idx(r, state=YState.DOWNLOADING) for r in
            (Resource.CPU, Resource.BANDWIDTH_IN, Resource.MEMORY)
        ]
        Cy_downloading_consumptions[scaled_resource_indices] *= scale_factor[None]

        scaled_resource_indices = [
            Resource.Y_resource_idx(r, state=YState.UPLOADING) for r in
            (Resource.CPU, Resource.BANDWIDTH_OUT, Resource.MEMORY)
        ]
        Cy_uploading_consumptions[scaled_resource_indices] *= scale_factor[None]

        scaled_resource_indices = [
            Resource.Y_resource_idx(r, state=YState.EXECUTING) for r in
            (Resource.CPU,)
        ]
        executing_scale_factor = scale_factor[None, None, :, None].expand(
            self._get_y_dimensions(YState.EXECUTING)).flatten()
        Cy_executing_consumptions[scaled_resource_indices] *= executing_scale_factor[None]

        for i in range(Cx_consumptions.shape[0]):
            Cx_consumptions[i] /= torch.clip(n_x_active_tasks, min=1.0)
        for i in range(Cy_downloading_consumptions.shape[0]):
            Cy_downloading_consumptions[i] /= torch.clip(
                n_y_downloading_tasks, min=1.0)
        for i in range(Cy_uploading_consumptions.shape[0]):
            Cy_uploading_consumptions[i] /= torch.clip(
                n_y_uploading_tasks, min=1.0)

        Cx_curr = torch.zeros_like(Cx_prev, dtype=Cx_prev.dtype)
        Cx_curr[...] = Cx_consumptions[:, None, None, :]
        Cx_curr[:, Xhat_curr != XState.DOWNLOADING] = 0

        online_mask = A_curr[Resource.BANDWIDTH_IN] > 0
        x_downloading_mask: torch.Tensor = Xhat_curr == XState.DOWNLOADING
        x_paused_mask: torch.Tensor = Xhat_curr == XState.PAUSED
        x_online_downloading_mask = online_mask[None,
                                                None, :] & x_downloading_mask
        x_ready_mask: torch.Tensor = Xhat_curr == XState.READY
        x_active_download_mask = x_downloading_mask | x_paused_mask

        self._storage_cumsum[x_online_downloading_mask] += Cx_curr[Resource.X_resource_idx(
            Resource.STORAGE)][x_online_downloading_mask]
        self._storage_cumsum[~x_active_download_mask] = 0.0

        storage_idx = Resource.X_resource_idx(Resource.STORAGE)

        Cx_curr[storage_idx] = self._storage_cumsum
        Cx_curr[storage_idx][x_ready_mask] = torch.repeat_interleave(
            self._model_context.partition_sizes[:, :, None],
            repeats=self._model_context.n_nodes,
            dim=2,
        )[x_ready_mask]

        Cy_curr = torch.zeros_like(Cy_prev)
        Cy_temp = torch.zeros_like(Cy_prev)

        downloading_res_idx = [
            Resource.Y_resource_idx(r) for r in Resource.Y_resources(state=YState.DOWNLOADING)
        ]

        Cy_temp[downloading_res_idx] = Cy_downloading_consumptions[:,
                                                                   None, None, :, None]
        Cy_temp[:, Yshat_curr != YState.DOWNLOADING] = 0
        Cy_curr += Cy_temp

        uploading_res_idx = [
            Resource.Y_resource_idx(r) for r in Resource.Y_resources(state=YState.UPLOADING)
        ]

        Cy_temp[uploading_res_idx] = Cy_uploading_consumptions[:,
                                                               None, None, :, None]
        Cy_temp[:, Yshat_curr != YState.UPLOADING] = 0
        Cy_curr += Cy_temp

        executing_res_idx = [
            Resource.Y_resource_idx(r) for r in Resource.Y_resources(state=YState.EXECUTING)
        ]

        Cy_temp[executing_res_idx] = Cy_executing_consumptions.view(
            (len(executing_res_idx), *Cy_curr.shape[1:]))
        Cy_temp[:, Yshat_curr != YState.EXECUTING] = 0
        Cy_curr += Cy_temp

        assert Cy_curr.isnan().sum() == 0
        assert Cy_curr.isinf().sum() == 0

        return dist.Delta(Cx_curr), dist.Delta(Cy_curr)

    def _get_x_consumptions(self, models, configs, simulators, Xhat_curr, A_curr):
        # Cx_prev, Cy_prev have shape [resource, partition, task_type, node]
        # Xhat_curr has shape [partition, task_type, node]
        # Yhat_curr has shape [partition, task_type, node, client_slot]
        # A_curr has shape [resource, node]

        online_mask = A_curr[Resource.BANDWIDTH_IN] > 0
        x_downloading_mask: torch.Tensor = Xhat_curr == XState.DOWNLOADING
        x_online_downloading_mask = online_mask[None,
                                                None, :] & x_downloading_mask

        online_downloading_node_mask = x_online_downloading_mask.any(
            dim=(0, 1))
        n_x_active_tasks = x_online_downloading_mask.sum(dim=(0, 1))

        curr_bandwidth_in_level = A_curr[Resource.BANDWIDTH_IN]

        # Determine means and stds for each resource based on weighted nearest neighbors given means and stds for different bandwidth levels
        for i, config in enumerate(configs):
            all_bandwidth_in_levels = torch.as_tensor(
                models[i].keys).flatten()

            weights = torch.zeros(
                (all_bandwidth_in_levels.shape[0], curr_bandwidth_in_level.shape[0]), dtype=curr_bandwidth_in_level.dtype)
            # Manhattan distance, the same as Euclidean in 1D space
            weights[:, online_downloading_node_mask] = 1 / \
                torch.abs(all_bandwidth_in_levels[:,
                                                  None] - curr_bandwidth_in_level[online_downloading_node_mask])
            weights_total = weights.sum(axis=0)

            means = torch.as_tensor([v for (_, v) in sorted(
                models[i].original_means.items(), key=lambda x: x[0])])
            stds = torch.as_tensor([v for (_, v) in sorted(
                models[i].original_stds.items(), key=lambda x: x[0])])
            mins = torch.as_tensor([v for (_, v) in sorted(
                models[i].original_mins.items(), key=lambda x: x[0])])

            means = torch.nan_to_num(means @ weights / weights_total, nan=0.0)
            stds = torch.nan_to_num(stds @ weights / weights_total, nan=1.0)
            mins = torch.nan_to_num(mins @ weights / weights_total, nan=0.0)

            assert torch.all(stds > 0.0)

            config.update(
                mean=means,
                std=stds,
                min=mins,
            )

        all_consumptions = torch.zeros(
            len(models), self._model_context.n_nodes)

        for i in range(len(models)):
            config = configs[i]
            resource_consumption = simulators[i].simulate(n_simulations=1, mask=online_downloading_node_mask,
                                                          config=config)
            all_consumptions[i] = resource_consumption[:, 0]

        return all_consumptions, n_x_active_tasks

    def _get_y_consumptions(
        self,
        downloading_models,
        executing_models,
        uploading_models,
        downloading_configs,
        executing_configs,
        uploading_configs,
        downloading_simulators,
        executing_simulators,
        uploading_simulators,
        Yshat_curr,
        Ty_prev,
        A_curr,
    ):
        y_downloading_mask: torch.Tensor = Yshat_curr == YState.DOWNLOADING
        y_executing_mask: torch.Tensor = Yshat_curr == YState.EXECUTING
        y_uploading_mask: torch.Tensor = Yshat_curr == YState.UPLOADING

        y_downloading_node_mask = y_downloading_mask.any(dim=(0, 1, 3))
        y_executing_node_mask = y_executing_mask.any(dim=(0, 1, 3))
        y_uploading_node_mask = y_uploading_mask.any(dim=(0, 1, 3))

        n_y_downloading_tasks = y_downloading_mask.sum(dim=(0, 1, 3))
        n_y_uploading_tasks = y_uploading_mask.sum(dim=(0, 1, 3))

        curr_bandwidth_in_levels = A_curr[Resource.BANDWIDTH_IN]

        # Determine means and stds for each resource based on weighted nearest neighbors given means and stds for different bandwidth levels
        for conf_idx, config in enumerate(downloading_configs):
            all_bandwidth_levels = torch.as_tensor(
                self._y_downloading_models[conf_idx].keys).flatten()

            weights = torch.zeros(
                (all_bandwidth_levels.shape[0], curr_bandwidth_in_levels.shape[0]), dtype=curr_bandwidth_in_levels.dtype)
            weights[:, y_downloading_node_mask] = 1 / torch.abs(
                all_bandwidth_levels[:, None] - curr_bandwidth_in_levels[y_downloading_node_mask])
            weights_total = weights.sum(axis=0)

            means = torch.as_tensor([v for (_, v) in sorted(
                downloading_models[conf_idx].original_means.items(), key=lambda x: x[0])])
            stds = torch.as_tensor([v for (_, v) in sorted(
                downloading_models[conf_idx].original_stds.items(), key=lambda x: x[0])])
            mins = torch.as_tensor([v for (_, v) in sorted(
                downloading_models[conf_idx].original_mins.items(), key=lambda x: x[0])])

            means = torch.nan_to_num(means @ weights / weights_total, nan=0.0)
            stds = torch.nan_to_num(stds @ weights / weights_total, nan=1.0)
            mins = torch.nan_to_num(mins @ weights / weights_total, nan=0.0)

            assert torch.all(stds > 0.0)

            config.update(
                mean=means,
                std=stds,
                min=mins,
            )

        curr_bandwidth_out_levels = A_curr[Resource.BANDWIDTH_OUT]

        # Determine means and stds for each resource based on weighted nearest neighbors given means and stds for different bandwidth levels
        for conf_idx, config in enumerate(uploading_configs):
            all_bandwidth_levels = torch.as_tensor(
                self._y_uploading_models[conf_idx].keys).flatten()

            weights = torch.zeros(
                (all_bandwidth_levels.shape[0], curr_bandwidth_out_levels.shape[0]), dtype=curr_bandwidth_out_levels.dtype)
            weights[:, y_uploading_node_mask] = 1 / torch.abs(
                all_bandwidth_levels[:, None] - curr_bandwidth_out_levels[y_uploading_node_mask])
            weights_total = weights.sum(axis=0)

            means = torch.as_tensor([v for (_, v) in sorted(
                uploading_models[conf_idx].original_means.items(), key=lambda x: x[0])])
            stds = torch.as_tensor([v for (_, v) in sorted(
                uploading_models[conf_idx].original_stds.items(), key=lambda x: x[0])])
            mins = torch.as_tensor([v for (_, v) in sorted(
                uploading_models[conf_idx].original_mins.items(), key=lambda x: x[0])])

            means = torch.nan_to_num(means @ weights / weights_total, nan=0.0)
            stds = torch.nan_to_num(stds @ weights / weights_total, nan=1.0)
            mins = torch.nan_to_num(mins @ weights / weights_total, nan=0.0)

            assert torch.all(stds > 0.0)

            config.update(
                mean=means,
                std=stds,
                min=mins,
            )

        y_executing_node_slot_mask = y_executing_mask.any(dim=(0, 1))

        # TODO: - just remove all modes that are not executing across all dimensions, instead of this complicated masking!

        Ty_executing = Ty_prev.clone()
        Ty_executing[..., ~y_executing_node_slot_mask] = -1
        Ty_executing = Ty_executing[:, :, y_executing_node_mask,
                                    :].view(-1, Ty_executing.shape[-1])

        Ty_executing_unique, Ty_executing_counts, Ty_executing_inverse, Ty_executing_first_occ_mask = unique_per_row(
            Ty_executing)

        curr_counts = torch.zeros_like(Ty_prev, dtype=torch.int32)
        curr_counts[:, :, y_executing_node_mask, :] = Ty_executing_counts.gather(
            dim=1, index=Ty_executing_inverse).view(curr_counts[:, :, y_executing_node_mask, :].shape)
        curr_counts = curr_counts.flatten()

        y_mask = torch.zeros_like(Yshat_curr, dtype=torch.bool)
        y_mask[:, :, y_executing_node_mask, :] = Ty_executing_first_occ_mask.view(
            y_mask[:, :, y_executing_node_mask, :].shape)
        y_mask = y_mask.flatten()

        y_executing_mask_flat = y_executing_mask.flatten()
        # Determine means and stds for each resource based on weighted nearest neighbors given means and stds for different task counts
        for conf_idx, config in enumerate(executing_configs):
            # TODO: - MOVE THIS LOGIC TO INIT SOMEWHERE (still relevant?)

            model_task_counts = defaultdict(list)
            for (task_count, model_name) in self._y_executing_models[conf_idx].keys:
                model_task_counts[model_name].append(task_count)

            model_task_counts = [model_task_counts[model]
                                 for model in self._model_context.model_list]
            for i in range(1, len(model_task_counts)):
                assert model_task_counts[i] == model_task_counts[0]
            task_counts = torch.as_tensor(
                model_task_counts[0], dtype=torch.int32)
            n_models = len(self._model_context.model_list)
            n_task_counts = task_counts.shape[0]

            weights = torch.zeros(
                (task_counts.shape[0], curr_counts.shape[0]), dtype=A_curr.dtype)
            weights[:, y_executing_mask_flat] = 1 / torch.abs(
                task_counts[:, None] - curr_counts[y_executing_mask_flat])

            # Handle infinite weights (exact matches)
            inf_idx = (weights == torch.inf).nonzero()
            weights[:, inf_idx[:, 1]] = 0.0
            weights[inf_idx[:, 0], inf_idx[:, 1]] = 1.0

            weights_total = weights.sum(axis=0)

            model_idx_map = {model: i for i, model in enumerate(
                self._model_context.model_list)}
            task_count_idx_map = {
                task_count.item(): i for i, task_count in enumerate(task_counts)}

            means = torch.zeros((n_models, n_task_counts), dtype=A_curr.dtype)
            stds = torch.zeros((n_models, n_task_counts), dtype=A_curr.dtype)
            mins = torch.zeros((n_models, n_task_counts), dtype=A_curr.dtype)

            means_items = list(
                executing_models[conf_idx].original_means.items())
            stds_items = list(
                executing_models[conf_idx].original_stds.items())
            mins_items = list(
                executing_models[conf_idx].original_mins.items())

            for (task_count, model_name), mean in means_items:
                model_idx = model_idx_map.get(model_name, -1)
                if model_idx != -1:
                    task_count_idx = task_count_idx_map[task_count]
                    means[model_idx, task_count_idx] = mean
            for (task_count, model_name), std in stds_items:
                model_idx = model_idx_map.get(model_name, -1)
                if model_idx != -1:
                    task_count_idx = task_count_idx_map[task_count]
                    stds[model_idx, task_count_idx] = std
            for (task_count, model_name), min_val in mins_items:
                model_idx = model_idx_map.get(model_name, -1)
                if model_idx != -1:
                    task_count_idx = task_count_idx_map[task_count]
                    mins[model_idx, task_count_idx] = min_val

            model_indices = self._model_context.partition_model_indices
            partition_mask = self._model_context.partition_mask

            partition_means = means[model_indices]
            partition_stds = stds[model_indices]
            partition_mins = mins[model_indices]
            partition_means[~partition_mask] = 0.0
            partition_stds[~partition_mask] = 0.0
            partition_mins[~partition_mask] = 0.0

            n_partitions = self._model_context.n_partitions
            n_tasks = self._model_context.n_tasks

            means = torch.nan_to_num(
                torch.einsum(
                    "ptn,nptc->ptc",
                    partition_means,
                    weights.view(n_task_counts, n_partitions, n_tasks, -1),
                ).view(-1) / weights_total,
                nan=0.0
            )
            stds = torch.nan_to_num(
                torch.einsum(
                    "ptn,nptc->ptc",
                    partition_stds,
                    weights.view(n_task_counts, n_partitions, n_tasks, -1),
                ).view(-1) / weights_total,
                nan=1.0
            )
            mins = torch.nan_to_num(
                torch.einsum(
                    "ptn,nptc->ptc",
                    partition_mins,
                    weights.view(n_task_counts, n_partitions, n_tasks, -1),
                ).view(-1) / weights_total,
                nan=0.0
            )

            assert torch.all(stds.view(n_partitions, n_tasks, -1)
                             [partition_mask] > 0.0)

            config.update(
                mean=means,
                std=stds,
                min=mins,
            )

        # Simulate resource consumptions for different states

        downloading_consumptions = torch.zeros(
            len(downloading_models), self._get_y_n_series(YState.DOWNLOADING))
        executing_consumptions = torch.zeros(
            len(executing_models), self._get_y_n_series(YState.EXECUTING))
        uploading_consumptions = torch.zeros(
            len(uploading_models), self._get_y_n_series(YState.UPLOADING))

        for i in range(len(downloading_models)):
            config = downloading_configs[i]
            resource_consumption = downloading_simulators[i].simulate(n_simulations=1, mask=y_downloading_node_mask,
                                                                      config=config)
            downloading_consumptions[i] = resource_consumption[:, 0]

        for i in range(len(executing_models)):
            config = executing_configs[i]
            resource_consumption = executing_simulators[i].simulate(n_simulations=1, mask=y_mask,
                                                                    config=config)

            if resource_consumption.isnan().sum() > 0:
                assert False

            resource_consumption = resource_consumption.reshape(
                self._get_y_dimensions(YState.EXECUTING))
            resource_consumption[:, :, y_executing_node_mask, :] = fill_all_occurrences(
                resource_consumption[:, :, y_executing_node_mask,
                                     :].view(-1, resource_consumption.shape[-1]),
                Ty_executing_unique,
                Ty_executing_inverse,
                Ty_executing_first_occ_mask,
            ).view(resource_consumption[:, :, y_executing_node_mask, :].shape)
            resource_consumption[~y_executing_mask] = 0
            resource_consumption = resource_consumption.flatten()

            executing_consumptions[i] = resource_consumption

        for i in range(len(uploading_models)):
            config = uploading_configs[i]
            resource_consumption = uploading_simulators[i].simulate(n_simulations=1, mask=y_uploading_node_mask,
                                                                    config=config)
            uploading_consumptions[i] = resource_consumption[:, 0]

        return downloading_consumptions, executing_consumptions, uploading_consumptions, n_y_downloading_tasks, curr_counts, n_y_uploading_tasks

    def _get_y_n_series(self, state):
        return prod(self._get_y_dimensions(state))

    def _get_y_dimensions(self, state):
        ctx = self._model_context

        if state == YState.EXECUTING:
            return (ctx.n_partitions, ctx.n_tasks, ctx.n_nodes, ctx.n_task_exec_slots)
        elif state == YState.DOWNLOADING or state == YState.UPLOADING:
            return (ctx.n_nodes,)
        else:
            raise ValueError(f"Invalid state for dimensions: {state}")

    def _get_x_simulators(self, models):
        return [
            VectorArimaSimulator(
                params=model.model_param_dict, n_series=self._model_context.n_nodes)
            for model in models
        ]

    def _get_y_simulators(self, models, state):
        n_series = self._get_y_n_series(state)

        return [
            VectorArimaSimulator(
                params=model.model_param_dict, n_series=n_series)
            for model in models
        ]

    def _get_x_configs(self, models):
        return [
            SimulatorConfig(
                n_series=self._model_context.n_nodes,
                min=torch.repeat_interleave(
                    torch.tensor([0.]), self._model_context.n_nodes),
                power_transform_lambda=model.power_transform_lambda,
            )
            for model in models
        ]

    def _get_y_configs(self, models, state):
        n_series = self._get_y_n_series(state)

        return [
            SimulatorConfig(
                n_series=n_series,
                min=torch.repeat_interleave(
                    torch.tensor([0.]), n_series),
                power_transform_lambda=model.power_transform_lambda,
            )
            for model in models
        ]
