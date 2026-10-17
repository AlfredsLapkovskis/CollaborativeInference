import pytest
import torch
import warnings
from itertools import product

from simulation.impl.c_transition_function_default import CTransitionFunctionDefault
from simulation.impl.a_transition_function_default import ATransitionFunctionDefault
from simulation.types import *
from simulation.model_context import ModelContext
from utils.resource_consumption_model import ResourceConsumptionModel
from utils.available_resources_model import AvailableResourcesModel


class TestCTransitionFunctionDefault:
    @pytest.fixture(
        params=[
            ModelContext(
                n_clients=1000,
                n_task_exec_slots=8,
                n_timesteps=1000,
                partition_sizes=torch.tensor([
                    [241745227, 57663892, 352978285, 347519341],
                    [574691033, 553449335, 81485780, 241745227],
                    [1226297693, 1220854109, 0, 0],
                ]),
                partition_models=[
                    ["resnet152", "densenet169", "vit-b-32", "vit-b-16"],
                    ["vgg19", "vgg16", "densenet201", "resnet152"],
                    ["vit-l-32", "vit-l-16"],
                ],
                task_query_probabilities=torch.tensor([0.5, 0.15, 0.25, 0.1]),
            ),
        ],
    )
    def model_context(self, request):
        return request.param

    # bandwidth in, cpu, memory, storage
    @pytest.fixture(
        params=[
            [
                ResourceConsumptionModel.load(
                    "resources/models/consumption_x_download_res_net_read_model.json"),
                ResourceConsumptionModel.load(
                    "resources/models/consumption_x_download_res_cpu_model.json"),
                ResourceConsumptionModel.load(
                    "resources/models/consumption_x_download_res_memory_model.json"),
                ResourceConsumptionModel.load(
                    "resources/models/consumption_x_download_res_storage_model.json"),
            ],
        ],
    )
    def x_resource_models(self, request):
        return request.param

    # bandwidth in, cpu, memory
    @pytest.fixture(
        params=[
            [
                ResourceConsumptionModel.load(
                    "resources/models/consumption_y_download_res_net_read_model.json"),
                ResourceConsumptionModel.load(
                    "resources/models/consumption_y_download_res_cpu_model.json"),
                ResourceConsumptionModel.load(
                    "resources/models/consumption_y_download_res_memory_model.json"),
            ],
        ],
    )
    def y_downloading_resource_models(self, request):
        return request.param

    # cpu, memory
    @pytest.fixture(
        params=[
            [
                ResourceConsumptionModel.load(
                    "resources/models/consumption_y_execution_res_cpu_model.json"),
                ResourceConsumptionModel.load(
                    "resources/models/consumption_y_execution_res_memory_model.json"),
            ],
        ],
    )
    def y_executing_resource_models(self, request):
        return request.param

    # bandwidth out, cpu, memory
    @pytest.fixture(
        params=[
            [
                ResourceConsumptionModel.load(
                    "resources/models/consumption_y_upload_res_net_write_model.json"),
                ResourceConsumptionModel.load(
                    "resources/models/consumption_y_upload_res_cpu_model.json"),
                ResourceConsumptionModel.load(
                    "resources/models/consumption_y_upload_res_memory_model.json"),
            ],
        ],
    )
    def y_uploading_resource_models(self, request):
        return request.param

    @pytest.fixture
    def c_transition_function_factory(
        self,
        model_context,
        x_resource_models,
        y_downloading_resource_models,
        y_executing_resource_models,
        y_uploading_resource_models
    ):
        return lambda: CTransitionFunctionDefault(
            model_context=model_context,
            x_resource_models=x_resource_models,
            y_downloading_resource_models=y_downloading_resource_models,
            y_executing_resource_models=y_executing_resource_models,
            y_uploading_resource_models=y_uploading_resource_models,
        )

    @pytest.fixture(
        params=[
            [
                AvailableResourcesModel.load(
                    "resources/models/available_bandwidth_in_model.json"),
                AvailableResourcesModel.load(
                    "resources/models/available_bandwidth_out_model.json"),
                AvailableResourcesModel.load(
                    "resources/models/available_cpu_model.json"),
                AvailableResourcesModel.load(
                    "resources/models/available_memory_model.json"),
                AvailableResourcesModel.load(
                    "resources/models/available_storage_model.json"),
            ],
        ],
    )
    def available_resource_models(self, request):
        return request.param

    @pytest.fixture(
        params=[
            # [924430038, 906977843, 9530.4450033333333, 3618423248, 1073741824000],
            [98415753, 98415753, 953.4450033333333, 3618423248, 275039403427],
        ],
        # ids=["server_normal", "server_small"],
    )
    def server_means(self, request):
        return request.param

    @pytest.fixture(
        params=[
            # [27432989, 29702961, 78.8521822691509, 468580277, 0.0],
            [2920543, 3189909, 78.8521822691509, 468580277, 2574260485],
        ],
        # ids=["server_normal", "server_small"],
    )
    def server_stds(self, request):
        return request.param

    @pytest.fixture(
        params=[
            # [1000000000, 1000000000, 2000.0, 32212254720, 1073741824000],
            [100000000, 100000000, 200.0, 3221225472, 3221225472],
        ],
        # ids=["server_normal", "server_small"],
    )
    def server_maxs(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [98415753, 98415753, 953.4450033333333, 3618423248, 275039403427],
        ],
    )
    def client_means(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [2920543, 3189909, 78.8521822691509, 468580277, 2574260485],
        ],
    )
    def client_stds(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [100000000, 100000000, 200.0, 3221225472, 3221225472],
        ],
    )
    def client_maxs(self, request):
        return request.param

    @pytest.fixture
    def a_transition_function_factory(
        self,
        model_context,
        available_resource_models,
        server_means,
        server_stds,
        server_maxs,
        client_means,
        client_stds,
        client_maxs,
    ):
        return lambda: ATransitionFunctionDefault(
            model_context=model_context,
            resource_models=available_resource_models,
            server_means=server_means,
            server_stds=server_stds,
            server_maxs=server_maxs,
            client_means=client_means,
            client_stds=client_stds,
            client_maxs=client_maxs,
        )

    def test_expected_output(
        self,
        c_transition_function_factory,
        a_transition_function_factory,
        model_context: ModelContext,
    ):
        x_shape = (model_context.n_partitions,
                   model_context.n_tasks, model_context.n_nodes)
        y_shape = (model_context.n_partitions, model_context.n_tasks,
                   model_context.n_nodes, model_context.n_task_exec_slots)
        cx_shape = (len(Resource.X_resources()), *x_shape)
        cy_shape = (len(Resource.Y_resources()), *y_shape)

        Cx_prev = torch.zeros(cx_shape, dtype=torch.float32)
        Cy_prev = torch.zeros(cy_shape, dtype=torch.float32)

        a_transition_function = a_transition_function_factory()
        c_transition_function = c_transition_function_factory()

        for iteration in range(1000):
            O = torch.randint(
                low=0, high=2, size=(model_context.n_clients,), dtype=torch.int8)
            A = a_transition_function(
                torch.zeros((len(Resource.available_resources()),
                            model_context.n_nodes)),
                O,
            ).sample()

            random_values = torch.randint(
                low=0, high=3, size=x_shape, dtype=torch.int8)

            x_online_mask = torch.ones(x_shape, dtype=torch.bool)
            x_online_mask[:, :, 1:] = (O == 1).view(1, 1, -1)

            Xhat = torch.full(x_shape, XState.EMPTY, dtype=torch.int8)
            Xhat[random_values == 0] = XState.EMPTY
            Xhat[random_values == 1] = XState.DOWNLOADING
            Xhat[random_values == 2] = XState.READY

            Xhat[:, :, 0] = XState.READY
            Xhat[~model_context.partition_mask] = XState.EMPTY

            Yihat = torch.randint(low=1, high=model_context.n_clients + 1,
                                  size=y_shape, dtype=torch.int32)
            Yihat[torch.randint(low=0, high=2, size=y_shape,
                                dtype=torch.bool)] = 0

            random_values = torch.randint(
                low=0, high=5, size=y_shape, dtype=torch.int8)

            y_online_mask = torch.ones(y_shape, dtype=torch.bool)
            y_online_mask[:, :, 1:] = (O == 1).view(1, 1, -1, 1)

            Yshat = torch.full(y_shape, YState.IDLE, dtype=torch.int8)
            Yshat[y_online_mask & (random_values == 0)] = YState.IDLE
            Yshat[y_online_mask & (random_values == 1)] = YState.DOWNLOADING
            Yshat[y_online_mask & (random_values == 2)] = YState.EXECUTING
            Yshat[y_online_mask & (random_values == 3)] = YState.UPLOADING
            Yshat[random_values == 4] = YState.DONE

            Yshat[Xhat != XState.READY] = YState.IDLE
            Yshat[Yihat == 0] = YState.IDLE
            Yihat[Yshat == YState.IDLE] = 0

            Tx = torch.randint(low=1, high=1_000,
                               size=x_shape, dtype=torch.int32)
            Tx[(Xhat == XState.EMPTY) | (Xhat == XState.READY)] = 0
            Tx[torch.randint(low=0, high=2, size=x_shape,
                             dtype=torch.bool)] = 0

            Ty = torch.randint(low=1, high=1_000,
                               size=y_shape, dtype=torch.int32)
            Ty[(Yshat == YState.IDLE) | (Yshat == YState.DONE)] = 0
            Ty[torch.randint(low=0, high=2, size=y_shape,
                             dtype=torch.bool)] = 0

            Cx_dist, Cy_dist = c_transition_function(
                Cx_prev, Cy_prev, Xhat, Yihat, Yshat, Tx, Ty, A)

            Cx = Cx_dist.sample()
            Cy = Cy_dist.sample()

            assert Cx.dtype == torch.float32
            assert Cy.dtype == torch.float32

            assert torch.all(~Cx.isnan())
            assert torch.all(~Cy.isnan())

            assert torch.all(Cx >= 0.)
            assert torch.all(Cy >= 0.)

            # Online/offline test

            x_resource_idx_without_storage = [
                Resource.X_resource_idx(r)
                for r in Resource.X_resources() if r != Resource.STORAGE
            ]
            x_storage_idx = Resource.X_resource_idx(Resource.STORAGE)

            Cx_without_storage = Cx[x_resource_idx_without_storage]
            assert torch.all(
                Cx_without_storage[~x_online_mask.expand_as(Cx_without_storage)] == 0.)

            assert torch.all(Cy[~y_online_mask.expand_as(Cy)] == 0.)

            # Cx tests

            assert torch.all(Cx[:, Xhat == XState.EMPTY] == 0.)
            assert torch.all(Cx[x_resource_idx_without_storage]
                             [:, Xhat == XState.READY] == 0.)

            Cx_storage = Cx[x_storage_idx]
            mask = (
                Xhat == XState.READY) & model_context.partition_mask[..., None]
            expected = model_context.partition_sizes[..., None].expand_as(mask)
            assert torch.all(Cx_storage[mask] == expected[mask])

            assert torch.all(Cx[:, Xhat == XState.DOWNLOADING] >= 0.)
            assert torch.all(
                Cx_storage[Xhat == XState.DOWNLOADING] <= expected[Xhat == XState.DOWNLOADING])

            Cx_downloading = Cx[:, Xhat == XState.DOWNLOADING]

            if Cx_downloading.numel() > 0:
                zero_fraction = (Cx_downloading == 0).sum(
                    dim=1) / Cx_downloading.numel()
                for res, zf in zip(Resource.X_resources(), zero_fraction):
                    if zf > 0.5:
                        warnings.warn(
                            f"{zf * 100}% of downloading states have zero consumption for {Resource.get_name(res)} at iteration {iteration}.")

            # Cy tests

            assert torch.all(Cy[:, Yshat == YState.IDLE] == 0.)
            assert torch.all(Cy[:, Yshat == YState.DONE] == 0.)

            y_resources = {
                Resource.Y_resource_idx(r)
                for r in Resource.Y_resources()
            }
            y_downloading_res_idx = {
                Resource.Y_resource_idx(r)
                for r in Resource.Y_resources(state=YState.DOWNLOADING)
            }
            y_executing_res_idx = {
                Resource.Y_resource_idx(r)
                for r in Resource.Y_resources(state=YState.EXECUTING)
            }
            y_uploading_res_idx = {
                Resource.Y_resource_idx(r)
                for r in Resource.Y_resources(state=YState.UPLOADING)
            }

            y_non_downloading_res_idx = list(y_resources.difference(
                y_downloading_res_idx))
            y_non_executing_res_idx = list(y_resources.difference(
                y_executing_res_idx))
            y_non_uploading_res_idx = list(y_resources.difference(
                y_uploading_res_idx))

            assert torch.all(Cy[y_non_downloading_res_idx]
                             [:, Yshat == YState.DOWNLOADING] == 0.)
            assert torch.all(Cy[y_non_executing_res_idx]
                             [:, Yshat == YState.EXECUTING] == 0.)
            assert torch.all(Cy[y_non_uploading_res_idx]
                             [:, Yshat == YState.UPLOADING] == 0.)

            Cy_downloading = Cy[list(y_downloading_res_idx)][:,
                                                             Yshat == YState.DOWNLOADING]
            if Cy_downloading.numel() > 0:
                zero_fraction = (Cy_downloading == 0).sum(
                    dim=1) / Cy_downloading.numel()
                for res, zf in zip(Resource.Y_resources(state=YState.DOWNLOADING), zero_fraction):
                    if zf > 0.5:
                        warnings.warn(
                            f"{zf * 100}% of downloading states have zero consumption for {Resource.get_name(res)} at iteration {iteration}.")

            Cy_executing = Cy[list(y_executing_res_idx)][:,
                                                         Yshat == YState.EXECUTING]
            if Cy_executing.numel() > 0:
                zero_fraction = (Cy_executing == 0).sum(
                    dim=1) / Cy_executing.numel()
                for res, zf in zip(Resource.Y_resources(state=YState.EXECUTING), zero_fraction):
                    if zf > 0.5:
                        warnings.warn(
                            f"{zf * 100}% of executing states have zero consumption for {Resource.get_name(res)} at iteration {iteration}.")

            Cy_uploading = Cy[list(y_uploading_res_idx)][:,
                                                         Yshat == YState.UPLOADING]
            if Cy_uploading.numel() > 0:
                zero_fraction = (Cy_uploading == 0).sum(
                    dim=1) / Cy_uploading.numel()
                for res, zf in zip(Resource.Y_resources(state=YState.UPLOADING), zero_fraction):
                    if zf > 0.5:
                        warnings.warn(
                            f"{zf * 100}% of uploading states have zero consumption for {Resource.get_name(res)} at iteration {iteration}.")

            Cx_prev = Cx
            Cy_prev = Cy

    def test_downloads_and_uploads_go_through_a_single_channel(
        self,
        c_transition_function_factory,
        a_transition_function_factory,
        model_context: ModelContext,
    ):
        x_shape = (model_context.n_partitions,
                   model_context.n_tasks, model_context.n_nodes)
        y_shape = (model_context.n_partitions, model_context.n_tasks,
                   model_context.n_nodes, model_context.n_task_exec_slots)
        cx_shape = (len(Resource.X_resources()), *x_shape)
        cy_shape = (len(Resource.Y_resources()), *y_shape)

        O = torch.ones((model_context.n_clients,), dtype=torch.int8)

        for i in range(10):
            A_prev = torch.zeros(
                (len(Resource.available_resources()), model_context.n_nodes), dtype=torch.float32)
            Cx_prev = torch.zeros(cx_shape, dtype=torch.float32)
            Cy_prev = torch.zeros(cy_shape, dtype=torch.float32)

            random_values = torch.randint(
                low=0, high=3, size=x_shape, dtype=torch.int8)

            Xhat = torch.full(x_shape, XState.EMPTY, dtype=torch.int8)
            Xhat[random_values == 0] = XState.EMPTY
            Xhat[random_values == 1] = XState.DOWNLOADING
            Xhat[random_values == 2] = XState.READY

            Xhat[:, :, 0] = XState.READY
            Xhat[~model_context.partition_mask] = XState.EMPTY

            Tx_prev = torch.randint(low=1, high=1_000,
                                    size=x_shape, dtype=torch.int32)
            Tx_prev[(Xhat == XState.EMPTY) | (Xhat == XState.READY)] = 0

            Yihat = torch.randint(low=1, high=model_context.n_clients + 1,
                                  size=y_shape, dtype=torch.int32)
            Yihat[torch.randint(low=0, high=2, size=y_shape,
                                dtype=torch.bool)] = 0

            random_values = torch.randint(
                low=0, high=5, size=y_shape, dtype=torch.int8)

            Yshat = torch.full(y_shape, YState.IDLE, dtype=torch.int8)
            Yshat[random_values == 0] = YState.IDLE
            Yshat[random_values == 1] = YState.DOWNLOADING
            Yshat[random_values == 2] = YState.EXECUTING
            Yshat[random_values == 3] = YState.UPLOADING
            Yshat[random_values == 4] = YState.DONE

            Yshat[Xhat != XState.READY] = YState.IDLE
            Yshat[Yihat == 0] = YState.IDLE
            Yihat[Yshat == YState.IDLE] = 0

            Ty_prev = torch.randint(low=1, high=1_000,
                                    size=y_shape, dtype=torch.int32)
            Ty_prev[(Yshat == YState.IDLE) | (Yshat == XState.READY)] = 0

            a_transition_function = a_transition_function_factory()
            c_transition_function = c_transition_function_factory()

            for t in range(100):
                A = a_transition_function(A_prev, O).sample()

                Cx_dist, Cy_dist = c_transition_function(
                    Cx_prev, Cy_prev, Xhat, Yihat, Yshat, Tx_prev, Ty_prev, A)

                Cx = Cx_dist.sample()
                Cy = Cy_dist.sample()

                x_storage_idx = Resource.X_resource_idx(Resource.STORAGE)

                storage_diff = Cx[x_storage_idx] - Cx_prev[x_storage_idx]
                storage_diff[Xhat != XState.DOWNLOADING] = -1.
                storage_diff = storage_diff.reshape(-1, storage_diff.shape[-1])

                max_storage_diff = storage_diff.max(dim=0, keepdim=True).values

                # Assert downloaded data only increases (or remains the same)
                assert torch.all((storage_diff >= 0) | (storage_diff == -1.))
                # Assert download happens through a single channel
                assert torch.all(
                    (storage_diff == max_storage_diff) | (storage_diff == -1.))

                n_downloading = (max_storage_diff != -1).sum()
                if n_downloading == 0:
                    warnings.warn(
                        f"At time {t} in iteration {i}, no downloading states found."
                    )
                else:
                    positive_percent = (max_storage_diff >
                                        0).sum() / n_downloading
                    if positive_percent < 0.5:
                        warnings.warn(
                            f"At time {t} in iteration {i}, only {positive_percent * 100}% of downloading states have positive storage increase."
                        )

                x_resource_idx_without_storage = [
                    Resource.X_resource_idx(r)
                    for r in Resource.X_resources() if r != Resource.STORAGE
                ]

                for res_id in x_resource_idx_without_storage:
                    x_resource_usage = Cx[res_id].clone()
                    x_resource_usage[Xhat != XState.DOWNLOADING] = -1.
                    x_resource_usage = x_resource_usage.reshape(
                        -1, x_resource_usage.shape[-1])
                    max_value = x_resource_usage.max(
                        dim=0, keepdim=True).values

                    assert torch.all((x_resource_usage == max_value) | (
                        x_resource_usage == -1.))

                    n_downloading = (max_value != -1).sum()
                    if n_downloading == 0:
                        warnings.warn(
                            f"At time {t} in iteration {i}, no downloading states found for X resource {resource}."
                        )
                    else:
                        positive_percent = (
                            max_value > 0).sum() / n_downloading
                        if positive_percent < 0.5:
                            resource = Resource.get_name(
                                Resource.X_resources()[res_id])
                            warnings.warn(
                                f"At time {t} in iteration {i}, only than {positive_percent * 100}% of downloading states have positive consumption for X resource {resource}."
                            )

                y_downloading_resource_idx = [Resource.Y_resource_idx(
                    r) for r in Resource.Y_resources(state=YState.DOWNLOADING)]

                for res_id in y_downloading_resource_idx:
                    y_resource_usage = Cy[res_id].clone()
                    y_resource_usage[Yshat != YState.DOWNLOADING] = -1.
                    y_resource_usage = y_resource_usage.reshape(
                        -1, y_resource_usage.shape[-2] * y_resource_usage.shape[-1])
                    max_value = y_resource_usage.max(
                        dim=0, keepdim=True).values

                    assert torch.all((y_resource_usage == max_value) | (
                        y_resource_usage == -1.))

                    n_downloading = (max_value != -1).sum()
                    if n_downloading == 0:
                        warnings.warn(
                            f"At time {t} in iteration {i}, no downloading states found for Y resource {resource}."
                        )
                    else:
                        positive_percent = (
                            max_value > 0).sum() / n_downloading
                        if positive_percent < 0.5:
                            resource = Resource.get_name(
                                Resource.Y_resources()[res_id])
                            warnings.warn(
                                f"At time {t} in iteration {i}, only {positive_percent * 100}% of downloading states have positive consumption for Y resource {resource}."
                            )

                y_uploading_resource_idx = [Resource.Y_resource_idx(
                    r) for r in Resource.Y_resources(state=YState.UPLOADING)]
                for res_id in y_uploading_resource_idx:
                    y_resource_usage = Cy[res_id].clone()
                    y_resource_usage[Yshat != YState.UPLOADING] = -1.
                    y_resource_usage = y_resource_usage.reshape(
                        -1, y_resource_usage.shape[-2] * y_resource_usage.shape[-1])
                    max_value = y_resource_usage.max(
                        dim=0, keepdim=True).values

                    assert torch.all((y_resource_usage == max_value) | (
                        y_resource_usage == -1.))

                    n_downloading = (max_value != -1).sum()
                    if n_downloading == 0:
                        warnings.warn(
                            f"At time {t} in iteration {i}, no uploading states found for Y resource {resource}."
                        )
                    else:
                        positive_percent = (
                            max_value > 0).sum() / n_downloading
                        if positive_percent < 0.5:
                            resource = Resource.get_name(
                                Resource.Y_resources()[res_id])
                            warnings.warn(
                                f"At time {t} in iteration {i}, less than {positive_percent * 100}% of uploading states have positive consumption for Y resource {resource}."
                            )
                A_prev = A
                Cx_prev = Cx
                Cy_prev = Cy
                Tx_prev = torch.where(Tx_prev > 0, Tx_prev + 1, Tx_prev)
                Ty_prev = torch.where(Ty_prev > 0, Ty_prev + 1, Ty_prev)

    def test_batched_execution(
        self,
        c_transition_function_factory,
        a_transition_function_factory,
        model_context: ModelContext,
    ):
        x_shape = (model_context.n_partitions,
                   model_context.n_tasks, model_context.n_nodes)
        y_shape = (model_context.n_partitions, model_context.n_tasks,
                   model_context.n_nodes, model_context.n_task_exec_slots)
        cx_shape = (len(Resource.X_resources()), *x_shape)
        cy_shape = (len(Resource.Y_resources()), *y_shape)

        O = torch.ones((model_context.n_clients,), dtype=torch.int8)
        Xhat = torch.full(x_shape, XState.READY, dtype=torch.int8)

        for i in range(50):
            A_prev = torch.zeros(
                (len(Resource.available_resources()), model_context.n_nodes), dtype=torch.float32)
            Cx_prev = torch.zeros(cx_shape, dtype=torch.float32)
            Cy_prev = torch.zeros(cy_shape, dtype=torch.float32)

            batches = self._generate_y_batches(model_context)

            Tx = torch.zeros(x_shape, dtype=torch.int32)
            Ty_prev = torch.zeros(y_shape, dtype=torch.int32)
            Yihat = torch.zeros(y_shape, dtype=torch.int32)
            Yshat = torch.full(y_shape, YState.IDLE, dtype=torch.int8)

            for idx, batch in enumerate(batches):
                t_value = idx  # + 1
                for indices in batch:
                    Ty_prev[*indices] = t_value
                    Yihat[*indices] = torch.randint(
                        low=1, high=model_context.n_clients + 1, size=(1,)).item()
                    Yshat[*indices] = YState.EXECUTING

            a_transition_function = a_transition_function_factory()
            c_transition_function = c_transition_function_factory()

            n_timesteps = 20

            for t in range(n_timesteps):
                A = a_transition_function(A_prev, O).sample()

                Cx_dist, Cy_dist = c_transition_function(
                    Cx_prev, Cy_prev, Xhat, Yihat, Yshat, Tx, Ty_prev, A)

                Cx = Cx_dist.sample()
                Cy = Cy_dist.sample()

                non_zero_count = 0
                batch_values = set()

                for batch in batches:
                    first_value = Cy[:, *batch[0]]
                    for indices in batch[1:]:
                        assert torch.all(first_value == Cy[:, *indices])
                    if t == (n_timesteps - 1) and torch.any(first_value != 0):
                        # prev_len = len(batch_values)
                        batch_values.add(str(first_value.tolist()))
                        # assert len(batch_values) > prev_len
                        non_zero_count += 1

                if non_zero_count > 0:
                    # Warn if most values are zeros
                    if non_zero_count < len(batches) / 2:
                        warnings.warn(
                            f"At time {t} in iteration {i}, only {non_zero_count}/{len(batch_values)} batch values are not all zeros."
                        )
                    # Warn if some batch values are not unique
                    if len(batch_values) != non_zero_count:
                        warnings.warn(
                            f"At time {t} in iteration {i}, some non-zero batch values are not unique (expected {non_zero_count} but got {len(batch_values)} unique values)."
                        )

                A_prev = A
                Cx_prev = Cx
                Cy_prev = Cy
                Ty_prev = torch.where(Ty_prev > 0, Ty_prev + 1, Ty_prev)

    def test_scaling_from_exceeding_cpu(
        self,
        c_transition_function_factory,
        a_transition_function_factory,
        model_context: ModelContext,
    ):
        # generate a normal sequence
        # then generate a sequence where A has very low CPU capacity
        # check that Cy is scaled up accordingly
        x_shape = (model_context.n_partitions,
                   model_context.n_tasks, model_context.n_nodes)
        y_shape = (model_context.n_partitions, model_context.n_tasks,
                   model_context.n_nodes, model_context.n_task_exec_slots)
        cx_shape = (len(Resource.X_resources()), *x_shape)
        cy_shape = (len(Resource.Y_resources()), *y_shape)

        O = torch.ones((model_context.n_clients,), dtype=torch.int8)

        for i in range(10):
            n_timesteps = 20

            A_trajectory = torch.zeros(
                (n_timesteps + 1, len(Resource.available_resources()), model_context.n_nodes), dtype=torch.float32)
            a_transition_function = a_transition_function_factory()
            for t in range(n_timesteps):
                A_trajectory[t +
                             1] = a_transition_function(A_trajectory[t], O).sample()

            random_values = torch.randint(
                low=0, high=3, size=x_shape, dtype=torch.int8)

            Xhat = torch.full(x_shape, XState.EMPTY, dtype=torch.int8)
            Xhat[random_values == 0] = XState.EMPTY
            Xhat[random_values == 1] = XState.DOWNLOADING
            Xhat[random_values == 2] = XState.READY

            Xhat[:, :, 0] = XState.READY
            Xhat[~model_context.partition_mask] = XState.EMPTY

            Yihat = torch.randint(low=1, high=model_context.n_clients + 1,
                                  size=y_shape, dtype=torch.int32)
            Yihat[torch.randint(low=0, high=2, size=y_shape,
                                dtype=torch.bool)] = 0

            random_values = torch.randint(
                low=0, high=5, size=y_shape, dtype=torch.int8)

            Yshat = torch.full(y_shape, YState.IDLE, dtype=torch.int8)
            Yshat[random_values == 0] = YState.IDLE
            Yshat[random_values == 1] = YState.DOWNLOADING
            Yshat[random_values == 2] = YState.EXECUTING
            Yshat[random_values == 3] = YState.UPLOADING
            Yshat[random_values == 4] = YState.DONE

            Yshat[Xhat != XState.READY] = YState.IDLE
            Yshat[Yihat == 0] = YState.IDLE
            Yihat[Yshat == YState.IDLE] = 0

            seed = torch.randint(low=0, high=1_000_0000, size=(1,)).item()

            def generate_C_trajectory(A_trajectory, seed):
                torch.manual_seed(seed)

                Tx_prev = torch.randint(low=1, high=1_000,
                                        size=x_shape, dtype=torch.int32)
                Tx_prev[(Xhat == XState.EMPTY) | (Xhat == XState.READY)] = 0

                Ty_prev = torch.randint(low=1, high=1_000,
                                        size=y_shape, dtype=torch.int32)
                Ty_prev[(Yshat == YState.IDLE) | (Yshat == XState.READY)] = 0

                Cx_prev = torch.zeros(cx_shape, dtype=torch.float32)
                Cy_prev = torch.zeros(cy_shape, dtype=torch.float32)

                Cx_trajectory = torch.zeros(
                    (n_timesteps, *cx_shape), dtype=torch.float32)
                Cy_trajectory = torch.zeros(
                    (n_timesteps, *cy_shape), dtype=torch.float32)

                c_transition_function = c_transition_function_factory()

                for t in range(n_timesteps):
                    A = A_trajectory[t + 1]

                    Cx_dist, Cy_dist = c_transition_function(
                        Cx_prev, Cy_prev, Xhat, Yihat, Yshat, Tx_prev, Ty_prev, A)

                    Cx = Cx_dist.sample()
                    Cy = Cy_dist.sample()

                    Cx_trajectory[t] = Cx
                    Cy_trajectory[t] = Cy

                    Cx_prev = Cx
                    Cy_prev = Cy
                    Tx_prev = torch.where(Tx_prev > 0, Tx_prev + 1, Tx_prev)
                    Ty_prev = torch.where(Ty_prev > 0, Ty_prev + 1, Ty_prev)

                return Cx_trajectory, Cy_trajectory

            Cx_trajectory_normal, Cy_trajectory_normal = generate_C_trajectory(
                A_trajectory, seed)

            x_cpu_idx = Resource.X_resource_idx(Resource.CPU)
            y_cpu_idx = Resource.Y_resource_idx(Resource.CPU)

            total_normal_cpu_consumption = (
                Cx_trajectory_normal[:, x_cpu_idx].sum(dim=(1, 2)) +
                Cy_trajectory_normal[:, y_cpu_idx].sum(dim=(1, 2, 4))
            )

            max_cpu = total_normal_cpu_consumption.mean() / 5

            A_trajectory_low_cpu = A_trajectory.clone()
            A_trajectory_low_cpu[:, Resource.CPU] = torch.clamp_max(
                A_trajectory_low_cpu[:, Resource.CPU], max=max_cpu,
            )

            scale_factor = torch.clip(A_trajectory_low_cpu[1:, Resource.CPU] /
                                      total_normal_cpu_consumption, max=1.0)
            scale_factor = torch.nan_to_num(scale_factor, nan=0.0)

            Cx_trajectory_low_cpu, Cy_trajectory_low_cpu = generate_C_trajectory(
                A_trajectory_low_cpu, seed)

            total_low_cpu_consumption = (
                Cx_trajectory_low_cpu[:, x_cpu_idx].sum(dim=(1, 2)) +
                Cy_trajectory_low_cpu[:, y_cpu_idx].sum(dim=(1, 2, 4))
            )

            assert torch.all((total_low_cpu_consumption -
                             A_trajectory_low_cpu[1:, Resource.CPU]) < 1e-4)

            assert torch.allclose(
                total_normal_cpu_consumption * scale_factor, total_low_cpu_consumption, atol=1e-5)

            # We skip testing STORAGE here for simplicity, because only its current increased is scaled, while its accumulated value is not
            scaled_x_resource_indices = [
                Resource.X_resource_idx(r)
                for r in {Resource.CPU, Resource.BANDWIDTH_IN}
            ]
            assert torch.allclose(
                (Cx_trajectory_normal[:, scaled_x_resource_indices] *
                 scale_factor[:, None, None, None, :]),
                Cx_trajectory_low_cpu[:, scaled_x_resource_indices],
                atol=1e-5,
            )

            # TODO: Should we really have MEMORY scaled? Maybe it's negligible enough to not matter
            scaled_y_downloading_resource_indices = [
                Resource.Y_resource_idx(r, state=YState.DOWNLOADING) for r in
                (Resource.CPU, Resource.BANDWIDTH_IN, Resource.MEMORY)
            ]
            assert torch.allclose(
                (Cy_trajectory_normal[:, scaled_y_downloading_resource_indices] *
                 scale_factor[:, None, None, None, :, None]),
                Cy_trajectory_low_cpu[:,
                                      scaled_y_downloading_resource_indices],
                atol=1e-5,
            )

            scaled_y_uploading_resource_indices = [
                Resource.Y_resource_idx(r, state=YState.UPLOADING) for r in
                (Resource.CPU, Resource.BANDWIDTH_OUT, Resource.MEMORY)
            ]
            assert torch.allclose(
                Cy_trajectory_normal[:, scaled_y_uploading_resource_indices] *
                scale_factor[:, None, None, None, :, None],
                Cy_trajectory_low_cpu[:, scaled_y_uploading_resource_indices],
                atol=1e-5,
            )

            scaled_y_executing_resource_indices = [
                Resource.Y_resource_idx(r, state=YState.EXECUTING) for r in
                (Resource.CPU,)
            ]
            assert torch.allclose(
                Cy_trajectory_normal[:, scaled_y_executing_resource_indices] *
                scale_factor[:, None, None, None, :, None],
                Cy_trajectory_low_cpu[:, scaled_y_executing_resource_indices],
                atol=1e-5,
            )

    def test_not_exceeding_certain_resource_limits(
        self,
        c_transition_function_factory,
        a_transition_function_factory,
        model_context: ModelContext,
    ):
        x_shape = (model_context.n_partitions,
                   model_context.n_tasks, model_context.n_nodes)
        y_shape = (model_context.n_partitions, model_context.n_tasks,
                   model_context.n_nodes, model_context.n_task_exec_slots)
        cx_shape = (len(Resource.X_resources()), *x_shape)
        cy_shape = (len(Resource.Y_resources()), *y_shape)

        O = torch.ones((model_context.n_clients,), dtype=torch.int8)

        n_timesteps = 100

        for i in range(10):
            A_trajectory = torch.zeros(
                (n_timesteps + 1, len(Resource.available_resources()), model_context.n_nodes), dtype=torch.float32)
            a_transition_function = a_transition_function_factory()
            for t in range(n_timesteps):
                A_trajectory[t +
                             1] = a_transition_function(A_trajectory[t], O).sample()

            # Test with different resource limits
            A_trajectory /= i + 1

            random_values = torch.randint(
                low=0, high=3, size=x_shape, dtype=torch.int8)

            Xhat = torch.full(x_shape, XState.EMPTY, dtype=torch.int8)
            Xhat[random_values == 0] = XState.EMPTY
            Xhat[random_values == 1] = XState.DOWNLOADING
            Xhat[random_values == 2] = XState.READY

            Xhat[:, :, 0] = XState.READY
            Xhat[~model_context.partition_mask] = XState.EMPTY

            Yihat = torch.randint(low=1, high=model_context.n_clients + 1,
                                  size=y_shape, dtype=torch.int32)
            Yihat[torch.randint(low=0, high=2, size=y_shape,
                                dtype=torch.bool)] = 0

            random_values = torch.randint(
                low=0, high=5, size=y_shape, dtype=torch.int8)

            Yshat = torch.full(y_shape, YState.IDLE, dtype=torch.int8)
            Yshat[random_values == 0] = YState.IDLE
            Yshat[random_values == 1] = YState.DOWNLOADING
            Yshat[random_values == 2] = YState.EXECUTING
            Yshat[random_values == 3] = YState.UPLOADING
            Yshat[random_values == 4] = YState.DONE

            Yshat_server = Yshat[:, :, 0, :]
            Yshat[:, :, 0, :][(Yshat_server == YState.DOWNLOADING) | (
                Yshat_server == YState.UPLOADING)] = YState.EXECUTING

            Yshat[Xhat != XState.READY] = YState.IDLE
            Yshat[Yihat == 0] = YState.IDLE
            Yihat[Yshat == YState.IDLE] = 0

            Tx_prev = torch.randint(low=1, high=1_000,
                                    size=x_shape, dtype=torch.int32)
            Tx_prev[(Xhat == XState.EMPTY) | (Xhat == XState.READY)] = 0

            Ty_prev = torch.randint(low=1, high=1_000,
                                    size=y_shape, dtype=torch.int32)
            Ty_prev[(Yshat == YState.IDLE) | (Yshat == XState.READY)] = 0

            Cx_prev = torch.zeros(cx_shape, dtype=torch.float32)
            Cy_prev = torch.zeros(cy_shape, dtype=torch.float32)

            Cx_trajectory = torch.zeros(
                (n_timesteps, *cx_shape), dtype=torch.float32)
            Cy_trajectory = torch.zeros(
                (n_timesteps, *cy_shape), dtype=torch.float32)

            c_transition_function = c_transition_function_factory()

            for t in range(n_timesteps):
                A = A_trajectory[t + 1]

                Cx_dist, Cy_dist = c_transition_function(
                    Cx_prev, Cy_prev, Xhat, Yihat, Yshat, Tx_prev, Ty_prev, A)

                Cx = Cx_dist.sample()
                Cy = Cy_dist.sample()

                Cx_trajectory[t] = Cx
                Cy_trajectory[t] = Cy

                Cx_prev = Cx
                Cy_prev = Cy
                Tx_prev = torch.where(Tx_prev > 0, Tx_prev + 1, Tx_prev)
                Ty_prev = torch.where(Ty_prev > 0, Ty_prev + 1, Ty_prev)

            C = torch.zeros_like(A_trajectory[1:])
            C[:, Resource.X_resources()] += Cx_trajectory.sum(dim=(2, 3))
            C[:, Resource.Y_resources()] += Cy_trajectory.sum(dim=(2, 3, 5))

            resources = [Resource.BANDWIDTH_IN,
                         Resource.BANDWIDTH_OUT, Resource.CPU]

            # Relative tolerance to account for float32 inaccuracies
            assert torch.all(C[:, resources] <=
                             A_trajectory[1:][:, resources] * 1.00001)

            assert torch.all(C[:, Resource.BANDWIDTH_IN, 0] == 0.)
            assert torch.all(C[:, Resource.BANDWIDTH_OUT, 0] == 0.)

            assert torch.all(C[:, Resource.BANDWIDTH_IN, 1:].sum(
                dim=1) <= A_trajectory[1:, Resource.BANDWIDTH_OUT, 0])
            assert torch.all(C[:, Resource.BANDWIDTH_OUT, 1:].sum(
                dim=1) <= A_trajectory[1:, Resource.BANDWIDTH_IN, 0])

    def _generate_y_batches(self, model_context: ModelContext):
        n_partitions = model_context.n_partitions
        n_tasks = model_context.n_tasks
        n_nodes = model_context.n_nodes
        n_exec_slots = model_context.n_task_exec_slots

        batches = []

        for (p, t, n) in product(
            range(n_partitions), range(n_tasks), range(n_nodes),
        ):
            if model_context.partition_mask[p, t] and torch.rand(1).item() > 0.5:
                temp_batches = [[] for _ in range(n_exec_slots)]

                for s in range(n_exec_slots):
                    if torch.rand(1).item() > 0.5:
                        idx = torch.randint(
                            low=0, high=n_exec_slots, size=(1,)).item()
                        temp_batches[idx].append((p, t, n, s))

                for batch in temp_batches:
                    if batch:
                        batches.append(batch)

        return batches
