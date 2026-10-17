import warnings
import pytest
import torch
from itertools import product

from simulation.model_context import ModelContext
from simulation.impl.ty_transition_function_default import TyTransitionFunctionDefault
from simulation.impl.c_transition_function_default import CTransitionFunctionDefault
from simulation.impl.a_transition_function_default import ATransitionFunctionDefault
from simulation.types import *
from utils.task_completion_model import TaskCompletionModel
from utils.available_resources_model import AvailableResourcesModel
from utils.resource_consumption_model import ResourceConsumptionModel


class TestTyTransitionFunctionDefault:
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

    @pytest.fixture(
        params=[
            [
                TaskCompletionModel.load(
                    "resources/models/task_completion_download_model.json"),
                TaskCompletionModel.load(
                    "resources/models/task_completion_execution_model.json"),
                TaskCompletionModel.load(
                    "resources/models/task_completion_upload_model.json"),
            ],
        ],
    )
    def task_completion_models(self, request):
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
            [924430038, 906977843, 9530.4450033333333, 3618423248, 1073741824000],
        ],
    )
    def server_means(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [27432989, 29702961, 78.8521822691509, 468580277, 0.0],
        ],
    )
    def server_stds(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [1000000000, 1000000000, 2000.0, 32212254720, 1073741824000],
        ],
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

    @pytest.fixture
    def ty_transition_function_factory(self, model_context, task_completion_models):
        return lambda: TyTransitionFunctionDefault(
            model_context=model_context,
            task_completion_models=task_completion_models,
        )

    def test_output(
        self,
        model_context: ModelContext,
        a_transition_function_factory,
        c_transition_function_factory,
        ty_transition_function_factory,
    ):
        a_shape = (len(Resource.available_resources()), model_context.n_nodes)
        x_shape = (model_context.n_partitions,
                   model_context.n_tasks, model_context.n_nodes)
        y_shape = (model_context.n_partitions, model_context.n_tasks,
                   model_context.n_nodes, model_context.n_task_exec_slots)
        cx_shape = (len(Resource.X_resources()), *x_shape)
        cy_shape = (len(Resource.Y_resources()), *y_shape)

        O = torch.ones((model_context.n_clients,), dtype=torch.int8)
        Xhat = torch.full(x_shape, fill_value=XState.READY, dtype=torch.int8)
        Tx = torch.zeros(x_shape, dtype=torch.int32)

        for i in range(1):
            A_prev = torch.zeros(a_shape, dtype=torch.float32)
            Cx_prev = torch.zeros(cx_shape, dtype=torch.float32)
            Cy_prev = torch.zeros(cy_shape, dtype=torch.float32)

            batches = self._generate_y_batches(model_context)

            Ty_prev = torch.zeros(y_shape, dtype=torch.int32)
            Yihat = torch.zeros(y_shape, dtype=torch.int32)
            Yshat = torch.full(
                y_shape, fill_value=YState.IDLE, dtype=torch.int8)

            random_values = torch.randint_like(Yshat, low=0, high=4)
            Yshat[random_values == 3] = YState.DONE
            Yihat[Yshat == YState.DONE] = torch.randint_like(
                Yihat[Yshat == YState.DONE], low=1, high=model_context.n_clients + 1)

            for idx, batch in enumerate(batches):
                t_value = (idx % model_context.n_task_exec_slots)  # + 1
                status = {
                    0: YState.EXECUTING,
                    1: YState.DOWNLOADING,
                    2: YState.UPLOADING,
                }[torch.randint(low=0, high=3, size=(1,)).item()]
                for indices in batch:
                    Ty_prev[*indices] = t_value
                    Yihat[*indices] = torch.randint(
                        low=1, high=model_context.n_clients + 1, size=(1,)).item()
                    Yshat[*indices] = status

            batch_state_update_timesteps = [""] * len(batches)

            n_timesteps = 50

            a_transition_function = a_transition_function_factory()
            c_transition_function = c_transition_function_factory()
            ty_transition_function = ty_transition_function_factory()

            updated_download_to_executing = False
            updated_executing_to_uploading = False
            updated_uploading_to_done = False

            for t in range(n_timesteps):
                print(f"Iteration {i}, timestep {t}")
                A = a_transition_function(A_prev, O).sample()
                Cx_dist, Cy_dist = c_transition_function(
                    Cx_prev, Cy_prev, Xhat, Yihat, Yshat, Tx, Ty_prev, A)
                Cx = Cx_dist.sample()
                Cy = Cy_dist.sample()

                Ty = ty_transition_function(Ty_prev, Yihat, Yshat, Cy).sample()

                assert torch.all(
                    Ty[(Yshat == YState.IDLE) | (Yshat == YState.DONE)] == 0)

                assert torch.all((Ty == 0) | (Ty == Ty_prev + 1))

                for idx, batch in enumerate(batches):
                    first_indices = batch[0]
                    yshat = Yshat[*first_indices]
                    ty = Ty[*first_indices]
                    ty_prev = Ty_prev[*first_indices]

                    for indices in batch[1:]:
                        assert Ty[*indices] == ty
                        assert Yihat[*indices] != 0
                        assert Yshat[*indices] == yshat
                    if ty == 0 and ty_prev != 0:
                        batch_state_update_timesteps[idx] += str(
                            ty_prev.item()) + ","

                download_to_executing_mask = (
                    (Yshat == YState.DOWNLOADING) & (Ty_prev != 0) & (Ty == 0))
                executing_to_uploading_mask = (
                    (Yshat == YState.EXECUTING) & (Ty_prev != 0) & (Ty == 0))
                uploading_to_done_mask = (
                    (Yshat == YState.UPLOADING) & (Ty_prev != 0) & (Ty == 0))

                updated_download_to_executing |= torch.any(
                    download_to_executing_mask).item()
                updated_executing_to_uploading |= torch.any(
                    executing_to_uploading_mask).item()
                updated_uploading_to_done |= torch.any(
                    uploading_to_done_mask).item()

                Yshat[download_to_executing_mask] = YState.EXECUTING
                Yshat[executing_to_uploading_mask] = YState.UPLOADING
                Yshat[uploading_to_done_mask] = YState.DONE

                A_prev = A
                Cx_prev = Cx
                Cy_prev = Cy
                Ty_prev = Ty

            assert updated_download_to_executing
            assert updated_executing_to_uploading
            assert updated_uploading_to_done

            batch_state_update_timesteps = set(batch_state_update_timesteps)

            if len(batches) != len(batch_state_update_timesteps):
                warnings.warn(
                    f"Some batches had identical state update timesteps: (expected {len(batches)}, got {len(batch_state_update_timesteps)} unique)")

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
