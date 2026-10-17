import torch
import pyro.distributions as dist

from simulation.policies import ClientPolicy
from simulation.types import *
from utils.unique import *


class ClientPolicyDefault(ClientPolicy):
    def __call__(self, O, To, Q, Tq, A, X, Tx, Cx, Yi, Ys, Ty, Cy, W):
        # A shape [resource, node]
        # Cx shape [resource, partition, task_type, node]
        # Cy shape [resource, partition, task_type, node, client]

        x_resources = Resource.X_resources()
        x_storage_idx = Resource.X_resource_idx(Resource.STORAGE)

        x_resources_without_storage = Resource.X_resources()
        x_resources_without_storage.remove(Resource.STORAGE)

        y_resources = Resource.Y_resources()
        y_downloading_resources = Resource.Y_resources(
            state=YState.DOWNLOADING)
        y_executing_resources = Resource.Y_resources(state=YState.EXECUTING)
        y_uploading_resources = Resource.Y_resources(state=YState.UPLOADING)

        y_executing_resource_idx = [Resource.Y_resource_idx(
            r,
        ) for r in y_executing_resources]
        y_executing_resource_idx_tensor = torch.as_tensor(
            y_executing_resource_idx, dtype=torch.long, device=Cy.device)

        offline_mask = O == 0
        x_downloading_mask = X == XState.DOWNLOADING
        x_paused_mask = X == XState.PAUSED

        y_downloading_mask = Ys == YState.DOWNLOADING
        y_executing_mask = Ys == YState.EXECUTING
        y_uploading_mask = Ys == YState.UPLOADING

        # [partition, task_type, client]
        X_actions = torch.full_like(X, fill_value=ClientXAction.NONE)
        Ys_actions = torch.full_like(Ys, fill_value=ClientYAction.NONE)
        Yi_actions = Yi.clone()

        # - Consider Cy but remember that Cx, Cy and A have different resources

        def get_resource_overflow(Cx, Cy, A):
            A_used = torch.zeros_like(A)
            A_used[x_resources] += Cx.sum(dim=(1, 2), dtype=Cx.dtype)
            A_used[y_resources] += Cy.sum(dim=(1, 2, 4), dtype=Cy.dtype)
            return torch.clip(A_used - A, min=0.)

        # shape [x_resource, node]
        resource_overflow = get_resource_overflow(Cx, Cy, A)
        resource_overflow_mask = resource_overflow > 0

        # Step 1:
        # Pause all downloads at this client if any non-storage resource is overflowing.
        # This releases non-storage pressure while preserving downloaded storage progress.
        non_storage_resource_overflow = resource_overflow_mask[x_resources_without_storage].any(dim=0)[
            None, None, :]
        X_actions[x_downloading_mask &
                  non_storage_resource_overflow] = ClientXAction.PAUSE

        # shape [partition, task_type, node]
        Cx_storage_modified = Cx[x_storage_idx].clone()
        Cx_storage_modified[X_actions == ClientXAction.ABORT] = 0

        # Step 2:
        # If storage is overflowing, remove a minimimal number of tasks to fix it
        shape2d = (-1, Cx_storage_modified.shape[-1])
        Cx_storage_modified_2d = Cx_storage_modified.view(shape2d)

        sort_idx = torch.argsort(
            Cx_storage_modified_2d, dim=0, descending=False)  # was descending=True
        unsort_idx = torch.argsort(sort_idx, dim=0)

        cumsums = torch.cumsum(torch.gather(
            Cx_storage_modified_2d, dim=0, index=sort_idx), dim=0)
        # shape [partition, task_type, client]
        abort_mask = cumsums > A[Resource.STORAGE][None]
        abort_mask = torch.gather(abort_mask, dim=0, index=unsort_idx)

        X_actions[abort_mask.view_as(X_actions)] = ClientXAction.ABORT
        # Make sure that server node cannot abort!
        X_actions[:, :, 0] = ClientXAction.NONE

        # Step 3: Abort downloading/executing Ys actions if the task partition is not ready or is aborted
        Ys_actions[((Ys == YState.DOWNLOADING) | (Ys == YState.EXECUTING)) & (
            (X[..., None] != XState.READY) | (X_actions[..., None] == ClientXAction.ABORT))] = ClientYAction.ABORT

        # Step 4:
        # In case there are still some resource overflows,
        # first abort downloading Y tasks,
        # then abort executing Y tasks,
        # and then uploading Y tasks.
        Cx_modified = Cx.clone()
        Cx_modified[:, X_actions == ClientXAction.ABORT] = 0

        Cx_paused_non_storage = Cx_modified[x_resources_without_storage]
        Cx_paused_non_storage[:, X_actions == ClientXAction.PAUSE] = 0
        Cx_modified[x_resources_without_storage] = Cx_paused_non_storage

        Cy_modified = Cy.clone()
        Cy_modified[:, Ys_actions == ClientYAction.ABORT] = 0
        resource_overflow = get_resource_overflow(Cx_modified, Cy_modified, A)
        resource_overflow_mask = resource_overflow > 0

        y_resource_overflow = resource_overflow_mask[y_downloading_resources].any(dim=0)[
            None, None, :, None]
        Ys_actions[y_downloading_mask &
                   y_resource_overflow] = ClientYAction.ABORT

        # - Ensure we consider only EXECUTING tasks in this segment

        Cy_modified[:, Ys_actions == ClientYAction.ABORT] = 0

        Ty_executing = Ty.clone()
        Ty_executing[Ys != YState.EXECUTING] = -1

        shape2d = (-1, Ty.shape[-1])
        Ty2d = Ty_executing.view(shape2d)

        Ty_unique, Ty_counts, Ty_inverse, Ty_first_occ_mask = unique_per_row(
            Ty2d)

        # Exclude sentinel value -1 (non-executing tasks) from first occurrence mask
        Ty_first_occ_mask = Ty_first_occ_mask & (Ty2d != -1)
        Ty_first_occ_tensor = Ty_first_occ_mask.view(Ty.shape)

        Cy_resources = Cy_modified.index_select(
            0, y_executing_resource_idx_tensor)
        # Make sure we consider only executing tasks
        Cy_resources[:, Ys != YState.EXECUTING] = 0

        Cy_resources *= Ty_counts.gather(dim=1,
                                         index=Ty_inverse).view(Ty.shape)[None]
        Cy_resources[:, ~Ty_first_occ_tensor] = 0

        Cy_consumption_level = (
            Cy_resources / A[y_executing_resources][:, None, None, :, None]).sum(dim=0)

        shape2d = (-1, Cy_consumption_level.shape[-2])
        shape_with_res = (Cy_resources.shape[0], -1, Cy_resources.shape[-2])

        transposed_shape = (
            *Cy_consumption_level.shape[:-2], Cy_consumption_level.shape[-1], Cy_consumption_level.shape[-2])

        sort_idx = torch.argsort(
            # Move slots before nodes for sorting
            Cy_consumption_level.transpose(2, 3).reshape(shape2d),
            dim=0,
            descending=False
        )[None]
        unsort_idx = torch.argsort(sort_idx, dim=1)

        def get_fixed_resources_for_executing_cumsums():
            Cy_fixed = Cy_modified.index_select(
                0, y_executing_resource_idx_tensor).clone()
            Cy_fixed[:, Ys == YState.EXECUTING] = 0

            Cy_fixed = Cy_fixed.sum(dim=(1, 2, 4))

            y_executing_resource_set = set(y_executing_resources)
            filtered_x_resources = [
                r for r in x_resources if r in y_executing_resource_set]

            x_resource_x_idx = [Resource.X_resource_idx(
                r) for r in filtered_x_resources]

            x_resource_y_idx = [Resource.Y_resource_idx(
                r,
                state=YState.EXECUTING
            ) for r in filtered_x_resources]

            Cx_consumptions = Cx_modified[x_resource_x_idx].sum(dim=(1, 2))
            Cy_fixed[x_resource_y_idx] += Cx_consumptions

            return Cy_fixed

        # shape [resource, partition, task_type, node, client]
        cumsums = torch.cumsum(torch.gather(
            Cy_resources.transpose(3, 4).reshape(shape_with_res),
            dim=1,
            index=sort_idx.expand(shape_with_res)
        ), dim=1)

        available_resources = torch.clip(
            A[y_executing_resources] - get_fixed_resources_for_executing_cumsums(), min=0)
        abort_mask = (cumsums > available_resources[:, None, :]).any(
            dim=0, keepdim=True)
        abort_mask = torch.gather(
            abort_mask, dim=1, index=unsort_idx)
        abort_mask = abort_mask.view(
            transposed_shape).transpose(2, 3).reshape_as(Ty2d)

        abort_mask = fill_all_occurrences(
            abort_mask, Ty_unique, Ty_inverse, Ty_first_occ_mask)
        abort_mask = abort_mask.view_as(Ty_executing)

        assert torch.all(
            Ys[abort_mask] == YState.EXECUTING)

        Ys_actions[abort_mask] = ClientYAction.ABORT

        Cy_modified[:, Ys_actions == ClientYAction.ABORT] = 0
        resource_overflow = get_resource_overflow(Cx_modified, Cy_modified, A)[
            y_uploading_resources]
        resource_overflow_mask = resource_overflow > 0

        y_resource_overflow = resource_overflow.any(dim=0)[
            None, None, :, None]
        Ys_actions[y_uploading_mask &
                   y_resource_overflow] = ClientYAction.ABORT

        # Recompute overflow after all Y-side abort decisions; this is used for resume checks.
        Cy_modified[:, Ys_actions == ClientYAction.ABORT] = 0
        resource_overflow = get_resource_overflow(Cx_modified, Cy_modified, A)
        resource_overflow_mask = resource_overflow > 0
        non_storage_resource_overflow = resource_overflow_mask[x_resources_without_storage].any(dim=0)[
            None, None, :]

        # Resume paused downloads when clients have no active Y tasks and no non-storage overflow.
        y_active_node_mask = (
            y_downloading_mask | y_executing_mask | y_uploading_mask
        ).any(dim=(0, 1, 3))
        resume_node_mask = (~y_active_node_mask) & (
            ~non_storage_resource_overflow.view(-1))
        X_actions[x_paused_mask &
                  (X_actions == ClientXAction.NONE) &
                  resume_node_mask[None, None, :]] = ClientXAction.RESUME

        # Make sure nothing happens to offline clients
        X_actions[:, :, 1:][:, :, offline_mask] = ClientXAction.NONE
        Ys_actions[:, :, 1:, :][:, :, offline_mask, :] = ClientYAction.NONE

        # Nullify Yi actions where Ys actions are ABORT
        Yi_actions[Ys_actions == ClientYAction.ABORT] = 0

        return dist.Delta(X_actions), dist.Delta(Yi_actions), dist.Delta(Ys_actions)
