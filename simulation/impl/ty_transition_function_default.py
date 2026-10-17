from torch.func import vmap
import pyro.distributions as dist
import torch

from simulation.model_context import ModelContext
from simulation.types import *
from simulation.transition_functions import TyTransitionFunction
from utils.task_completion_model import TaskCompletionModel
from utils.vector_plr_simulator import VectorPLRSimulator
from utils.unique import *


class TyTransitionFunctionDefault(TyTransitionFunction):
    def __init__(
        self,
        model_context: ModelContext,
        task_completion_models: list[TaskCompletionModel],
    ):
        assert len(
            task_completion_models) == 3, "We need 3 task completion models: downloading, executing, uploading"

        self._model_context = model_context
        self._models = task_completion_models

        n_series = model_context.n_partitions * model_context.n_tasks * \
            model_context.n_nodes * model_context.n_task_exec_slots  # We need n_task_exec_slots because there might be several batches of tasks running asynchrnonously on the same node
        self._simulator = VectorPLRSimulator(  # TODO: - fix
            task_completion_models=task_completion_models)

        strata = task_completion_models[0].model_strata
        strata_idx = torch.zeros(
            (model_context.n_partitions, model_context.n_tasks), dtype=torch.int32)
        strata_map = {name: idx for idx, name in enumerate(strata)}
        for p in range(model_context.n_partitions):
            for t in range(model_context.n_tasks):
                if model_context.partition_mask[p, t]:
                    model_name = model_context.partition_models[p][t]
                    strata_idx[p, t] = strata_map[model_name]
        strata_idx = strata_idx[:, :, None, None].expand(
            model_context.n_partitions,
            model_context.n_tasks,
            model_context.n_nodes,
            model_context.n_task_exec_slots
        )
        self._strata_idx = strata_idx.contiguous()

    def __call__(self, Ty_prev, Yihat_curr, Yshat_curr, Cy_curr):
        # Ty_prev, Yhat_curr have shape [partition, task_type, node, client_slot]
        # Cy_curr has shape [y_resource, partition, task_type, node]
        # A_curr has shape [resource, node]

        resource, partition, task_type, node, client_slot = Cy_curr.shape
        Ty_vector_shape = (partition * task_type * node * client_slot,)
        Ty_matrix_shape = (partition * task_type * node, client_slot)
        Cy_2d_shape = (resource, partition * task_type * node * client_slot)
        Cy_3d_shape = (resource, partition * task_type * node, client_slot)

        Ty_counts_2d = torch.zeros(Ty_matrix_shape, dtype=Ty_prev.dtype)
        Cy_3d = torch.zeros(Cy_3d_shape, dtype=Cy_curr.dtype)
        model_2d = torch.zeros(Ty_matrix_shape, dtype=torch.int32)
        mask_2d = torch.zeros(Ty_matrix_shape, dtype=torch.bool)

        def fill_and_get_params(status):
            slot_mask = (Yshat_curr == status).view(-1, Yshat_curr.shape[-1])

            model_2d[slot_mask] = {
                YState.DOWNLOADING: 0,
                YState.EXECUTING: 1,
                YState.UPLOADING: 2,
            }[status]

            node_mask = slot_mask.any(dim=1)
            # just using any downloading/executing/uploading is not sufficient, because it will mix different statuses together
            slot_mask = slot_mask[node_mask]

            full_mask = torch.zeros_like(Ty_counts_2d, dtype=torch.bool)
            full_mask[node_mask] = slot_mask

            Ty_status = Ty_prev.view(Ty_matrix_shape)[node_mask]
            Ty_status[~slot_mask] = -1

            Ty_status_unique, Ty_status_counts, Ty_status_inverse, Ty_status_first_occ_mask = unique_per_row(
                Ty_status)

            Ty_counts_2d[full_mask] = Ty_status_counts.gather(
                dim=1, index=Ty_status_inverse)[slot_mask]
            Cy_3d[:, full_mask] = Cy_curr.view(
                Cy_3d.shape)[:, full_mask]
            mask_2d[full_mask] = Ty_status_first_occ_mask[slot_mask]

            return Ty_status_unique, Ty_status_inverse, Ty_status_first_occ_mask, full_mask, node_mask, slot_mask

        Ty_downloading_unique, Ty_downloading_inverse, Ty_downloading_first_occ_mask, downloading_full_mask, downloading_node_mask, downloading_slot_mask = fill_and_get_params(
            YState.DOWNLOADING)
        Ty_executing_unique, Ty_executing_inverse, Ty_executing_first_occ_mask, executing_full_mask, executing_node_mask, executing_slot_mask = fill_and_get_params(
            YState.EXECUTING)
        Ty_uploading_unique, Ty_uploading_inverse, Ty_uploading_first_occ_mask, uploading_full_mask, uploading_node_mask, uploading_slot_mask = fill_and_get_params(
            YState.UPLOADING)

        task_counts = Ty_counts_2d.view(Ty_vector_shape)[None, :]
        resource_consumptions = Cy_3d.view(Cy_2d_shape) * task_counts

        survival_probabilities = self._simulator.simulate(
            const_covariates=task_counts,
            var_covariates=resource_consumptions,
            time=Ty_prev.view(Ty_vector_shape),
            model_idx=model_2d.view(Ty_vector_shape),
            strata_idx=self._strata_idx.view(Ty_vector_shape),
            mask=mask_2d.view(Ty_vector_shape),
        ).reshape(Ty_prev.shape)

        Ty_increment = Ty_prev + 1
        Ty_reset = torch.zeros_like(Ty_prev)

        outcomes = torch.stack([Ty_reset, Ty_increment], dim=-1)
        probs = torch.stack([1 - survival_probabilities,
                            survival_probabilities], dim=-1)

        distribution: dist.Distribution = dist.MixtureSameFamily(
            mixture_distribution=dist.Categorical(probs=probs),
            component_distribution=dist.Delta(outcomes)
        )

        Ty_curr_matrix = distribution.sample().reshape(Ty_matrix_shape)

        Ty_curr_matrix[downloading_full_mask] = fill_all_occurrences(
            Ty_curr_matrix[downloading_node_mask],
            Ty_downloading_unique,
            Ty_downloading_inverse,
            Ty_downloading_first_occ_mask,
        )[downloading_slot_mask]
        Ty_curr_matrix[executing_full_mask] = fill_all_occurrences(
            Ty_curr_matrix[executing_node_mask],
            Ty_executing_unique,
            Ty_executing_inverse,
            Ty_executing_first_occ_mask,
        )[executing_slot_mask]
        Ty_curr_matrix[uploading_full_mask] = fill_all_occurrences(
            Ty_curr_matrix[uploading_node_mask],
            Ty_uploading_unique,
            Ty_uploading_inverse,
            Ty_uploading_first_occ_mask,
        )[uploading_slot_mask]

        Ty_curr = Ty_curr_matrix.reshape(Ty_prev.shape)

        # To fix non-incrementing states
        Ty_curr[(Yshat_curr == YState.IDLE) | (Yshat_curr == YState.DONE)] = 0

        return dist.Delta(Ty_curr)
