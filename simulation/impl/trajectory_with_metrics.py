import torch
import fastdigest

from simulation.model_context import ModelContext
from simulation.trajectory import Trajectory
from simulation.types import *


class TrajectoryWithMetrics(Trajectory):
    def __init__(self, model_context: ModelContext):
        self._model_context = model_context
        self._n_tasks = model_context.n_tasks
        self._n_partitions = model_context.n_partitions

        def get_empty_query_array():
            return [[] for _ in range(self._n_tasks)]

        # How about having here both cumulative and non-cumulative metrics?

        self.n_online_clients = []
        self.n_offline_clients = []

        self.active_queries = get_empty_query_array()
        self.cum_active_queries = get_empty_query_array()

        # Cumulative
        self.finished_queries = get_empty_query_array()
        self.cancelled_queries = get_empty_query_array()

        self.total_latency = get_empty_query_array()
        self.mean_latency = get_empty_query_array()
        self.std_latency = get_empty_query_array()
        self.min_latency = get_empty_query_array()
        self.max_latency = get_empty_query_array()

        self.cum_total_latency = get_empty_query_array()
        self.cum_mean_latency = get_empty_query_array()
        self.cum_std_latency = get_empty_query_array()
        self.cum_min_latency = get_empty_query_array()
        self.cum_max_latency = get_empty_query_array()

        self.p50_latency = get_empty_query_array()
        self.p95_latency = get_empty_query_array()
        self.p99_latency = get_empty_query_array()
        self.p999_latency = get_empty_query_array()

        self.server_bandwidth_in_usage = []
        self.server_bandwidth_out_usage = []
        self.server_cpu_usage = []
        self.server_memory_usage = []

        # Only online clients considered

        self.online_client_total_bandwidth_in_usage = []
        self.online_client_total_bandwidth_out_usage = []
        self.online_client_total_cpu_usage = []
        self.online_client_total_memory_usage = []
        self.online_client_total_storage_usage = []

        self.online_client_mean_bandwidth_in_usage = []
        self.online_client_mean_bandwidth_out_usage = []
        self.online_client_mean_cpu_usage = []
        self.online_client_mean_memory_usage = []
        self.online_client_mean_storage_usage = []

        self.online_client_std_bandwidth_in_usage = []
        self.online_client_std_bandwidth_out_usage = []
        self.online_client_std_cpu_usage = []
        self.online_client_std_memory_usage = []
        self.online_client_std_storage_usage = []

        self.offline_client_total_storage_usage = []
        self.offline_client_mean_storage_usage = []
        self.offline_client_std_storage_usage = []

        # self.client_min_bandwidth_in_usage = []
        # self.client_min_bandwidth_out_usage = []
        # self.client_min_cpu_usage = []
        # self.client_min_memory_usage = []
        # self.client_min_storage_usage = []

        # self.client_max_bandwidth_in_usage = []
        # self.client_max_bandwidth_out_usage = []
        # self.client_max_cpu_usage = []
        # self.client_max_memory_usage = []
        # self.client_max_storage_usage = []

        # Below metrics assume that subtasks of the same type for the same exact query are scheduled maximum once at the same time.

        self.required_total_y_subtasks = get_empty_query_array()

        self.server_total_executing_y_subtasks = get_empty_query_array()
        self.server_total_done_y_subtasks = get_empty_query_array()

        self.online_client_total_downloading_x_subtasks = get_empty_query_array()
        self.online_client_total_paused_x_subtasks = get_empty_query_array()
        self.online_client_total_ready_x_subtasks = get_empty_query_array()
        self.online_client_total_downloading_y_subtasks = get_empty_query_array()
        self.online_client_total_executing_y_subtasks = get_empty_query_array()
        self.online_client_total_uploading_y_subtasks = get_empty_query_array()
        self.online_client_total_done_y_subtasks = get_empty_query_array()

        self.online_client_mean_downloading_x_subtasks = get_empty_query_array()
        self.online_client_mean_paused_x_subtasks = get_empty_query_array()
        self.online_client_mean_ready_x_subtasks = get_empty_query_array()
        self.online_client_mean_downloading_y_subtasks = get_empty_query_array()
        self.online_client_mean_executing_y_subtasks = get_empty_query_array()
        self.online_client_mean_uploading_y_subtasks = get_empty_query_array()
        self.online_client_mean_done_y_subtasks = get_empty_query_array()

        self.online_client_std_downloading_x_subtasks = get_empty_query_array()
        self.online_client_std_paused_x_subtasks = get_empty_query_array()
        self.online_client_std_ready_x_subtasks = get_empty_query_array()
        self.online_client_std_downloading_y_subtasks = get_empty_query_array()
        self.online_client_std_executing_y_subtasks = get_empty_query_array()
        self.online_client_std_uploading_y_subtasks = get_empty_query_array()
        self.online_client_std_done_y_subtasks = get_empty_query_array()

        self.offline_client_total_downloading_x_subtasks = get_empty_query_array()
        self.offline_client_mean_downloading_x_subtasks = get_empty_query_array()
        self.offline_client_std_downloading_x_subtasks = get_empty_query_array()
        self.offline_client_total_paused_x_subtasks = get_empty_query_array()
        self.offline_client_mean_paused_x_subtasks = get_empty_query_array()
        self.offline_client_std_paused_x_subtasks = get_empty_query_array()
        self.offline_client_total_ready_x_subtasks = get_empty_query_array()
        self.offline_client_mean_ready_x_subtasks = get_empty_query_array()
        self.offline_client_std_ready_x_subtasks = get_empty_query_array()

        # We can aggregate the above task metrics across all task types, but std requires special handling (we can't just sum stds in general).
        self.online_client_std_downloading_x_subtasks_all = []
        self.online_client_std_paused_x_subtasks_all = []
        self.online_client_std_ready_x_subtasks_all = []
        self.online_client_std_downloading_y_subtasks_all = []
        self.online_client_std_executing_y_subtasks_all = []
        self.online_client_std_uploading_y_subtasks_all = []
        self.online_client_std_done_y_subtasks_all = []
        self.offline_client_std_downloading_x_subtasks_all = []
        self.offline_client_std_paused_x_subtasks_all = []
        self.offline_client_std_ready_x_subtasks_all = []

        # self.client_min_downloading_x_subtasks = get_empty_query_array()
        # self.client_min_downloading_y_subtasks = get_empty_query_array()
        # self.client_min_executing_y_subtasks = get_empty_query_array()
        # self.client_min_uploading_y_subtasks = get_empty_query_array()

        # self.client_max_downloading_x_subtasks = get_empty_query_array()
        # self.client_max_downloading_y_subtasks = get_empty_query_array()
        # self.client_max_executing_y_subtasks = get_empty_query_array()
        # self.client_max_uploading_y_subtasks = get_empty_query_array()

        self.cum_server_y_subtasks_completed = get_empty_query_array()
        self.cum_server_y_subtasks_aborted = get_empty_query_array()

        self.cum_client_x_subtasks_completed = get_empty_query_array()
        self.cum_client_x_subtasks_removed = get_empty_query_array()
        self.cum_client_x_subtasks_aborted = get_empty_query_array()
        self.cum_client_y_subtasks_completed = get_empty_query_array()
        self.cum_client_y_subtasks_aborted = get_empty_query_array()

        self.server_constraints_violation = []
        self.n_client_constraints_violations = []

        self._t_digests = [fastdigest.TDigest(
            max_centroids=1000) for _ in range(self._n_tasks)]
        self._timestep = 0
        self._prev_Q = None
        self._prev_Tq = None
        self._finished_indices = [torch.empty(
            0, dtype=torch.int64) for _ in range(self._n_tasks)]

    def append(
        self,
        O: torch.Tensor,
        To: torch.Tensor,
        Q: torch.Tensor,
        Tq: torch.Tensor,
        A: torch.Tensor,
        Xhat: torch.Tensor,
        Cx: torch.Tensor,
        Tx: torch.Tensor,
        X: torch.Tensor,
        Yihat: torch.Tensor,
        Yshat: torch.Tensor,
        Cy: torch.Tensor,
        Ty: torch.Tensor,
        Yi: torch.Tensor,
        Ys: torch.Tensor,
        W: torch.Tensor,
        server_X_action: torch.Tensor,
        client_X_action: torch.Tensor,
        server_Yi_action: torch.Tensor,
        client_Yi_action: torch.Tensor,
        server_Ys_action: torch.Tensor,
        client_Ys_action: torch.Tensor,
    ):
        # Online/offline clients

        self.n_online_clients.append(O.sum().item())
        self.n_offline_clients.append((O == 0).sum().item())

        def _safe_std(values: torch.Tensor) -> float:
            if values.numel() == 0:
                return 0.0
            return values.to(torch.float32).std(correction=0).nan_to_num(nan=0.0).item()

        # Queries

        for t in range(self._n_tasks):
            q = t + 1

            self.active_queries[t].append((Q == q).sum().item())
            if self._timestep > 0:
                self.cum_active_queries[t].append(
                    self.cum_active_queries[t][-1] + ((Q == q) & (self._prev_Q == 0)).sum().item())
            else:
                self.cum_active_queries[t].append(self.active_queries[t][-1])

            prevQ = self._prev_Q
            if prevQ is not None:
                active_q_indices = 1 + torch.arange(
                    len(Q), device=Q.device, dtype=torch.long)[Q == q]
                ended_q_indices = 1 + torch.arange(
                    len(Q), device=Q.device, dtype=torch.long)[(Q == 0) & (prevQ == q)]

                finished_mask = torch.ones_like(
                    active_q_indices, dtype=torch.bool)
                for p in range(self._n_partitions):
                    if not self._model_context.partition_mask[p, t]:
                        continue

                    yi = Yi[p, t][server_Ys_action[p, t]
                                  == ServerYAction.FINISH].to(active_q_indices.device)
                    finished_mask &= torch.isin(active_q_indices, yi)
                self.finished_queries[t].append(finished_mask.sum().item())

                finished_indices = self._finished_indices[t]
                if finished_indices.device != ended_q_indices.device:
                    finished_indices = finished_indices.to(
                        ended_q_indices.device)

                cancelled_mask = ~torch.isin(
                    ended_q_indices, finished_indices)
                self.cancelled_queries[t].append(
                    cancelled_mask.sum().item())

                self._finished_indices[t] = active_q_indices[finished_mask]

                # indices = 1 + torch.arange(len(Q))[(Q == 0) & (prevQ == q)]
                # finished = torch.ones_like(indices, dtype=torch.bool)

                # for p in range(self._n_partitions):
                #     if not self._model_context.partition_mask[p, t]:
                #         continue

                #     yi = Yi[p, t][Ys[p, t] == YState.DONE]
                #     finished &= torch.isin(indices, yi)

                # self.cum_finished_queries[t].append(finished.sum().item())
                # self.cum_cancelled_queries[t].append((~finished).sum().item())
            else:
                self.finished_queries[t].append(0)
                self.cancelled_queries[t].append(0)

        # Latencies

        for t in range(self._n_tasks):
            q = t + 1

            tq = Tq[Q == q].to(torch.float32)

            # Current latencies

            self.total_latency[t].append(tq.sum().item())
            self.mean_latency[t].append(
                tq.mean().item() if tq.numel() > 0 else 0.)
            self.std_latency[t].append(_safe_std(tq))
            self.min_latency[t].append(
                tq.min().item() if tq.numel() > 0 else 0.)
            self.max_latency[t].append(
                tq.max().item() if tq.numel() > 0 else 0.)

            # Cumulative latencies

            if self._timestep > 0:
                self.cum_total_latency[t].append(
                    self.cum_total_latency[t][-1] + self.total_latency[t][-1])

                last_cum_active_queries = self.cum_active_queries[t][-1]
                if last_cum_active_queries > 0:
                    self.cum_mean_latency[t].append(self.cum_mean_latency[t][-1] + (
                        self.total_latency[t][-1] - self.cum_mean_latency[t][-1]) / last_cum_active_queries)
                    self.cum_std_latency[t].append((self.cum_std_latency[t][-1] + (
                        self.total_latency[t][-1] - self.cum_mean_latency[t][-2]) * (self.total_latency[t][-1] - self.cum_mean_latency[t][-1]) / last_cum_active_queries) ** 0.5)
                else:
                    self.cum_mean_latency[t].append(0.)
                    self.cum_std_latency[t].append(0.)

                self.cum_min_latency[t].append(
                    min(self.cum_min_latency[t][-1], self.min_latency[t][-1]))
                self.cum_max_latency[t].append(
                    max(self.cum_max_latency[t][-1], self.max_latency[t][-1]))
            else:
                self.cum_total_latency[t].append(self.total_latency[t][-1])
                self.cum_mean_latency[t].append(self.mean_latency[t][-1])
                self.cum_std_latency[t].append(self.std_latency[t][-1])
                self.cum_min_latency[t].append(self.min_latency[t][-1])
                self.cum_max_latency[t].append(self.max_latency[t][-1])

        # Px latencies

        def percentile_or_0(t, p):
            try:
                return self._t_digests[t].percentile(p)
            except ValueError:
                return 0.

        prevTq = self._prev_Tq
        prevQ = self._prev_Q
        for t in range(self._n_tasks):
            if prevTq is not None and prevQ is not None:
                q = t + 1
                tq = prevTq[(prevQ == q) & (Q == 0)]

                if tq.numel() > 0:
                    self._t_digests[t].batch_update(tq.tolist())

                self.p50_latency[t].append(percentile_or_0(t, 50))
                self.p95_latency[t].append(percentile_or_0(t, 95))
                self.p99_latency[t].append(percentile_or_0(t, 99))
                self.p999_latency[t].append(percentile_or_0(t, 99.9))
            else:
                self.p50_latency[t].append(0.)
                self.p95_latency[t].append(0.)
                self.p99_latency[t].append(0.)
                self.p999_latency[t].append(0.)

        # Server resources

        self.server_bandwidth_in_usage.append(
            Cy[Resource.Y_resource_idx(
                Resource.BANDWIDTH_OUT), :, :, 1:, :].sum().item())
        self.server_bandwidth_out_usage.append(
            (
                Cx[Resource.X_resource_idx(Resource.BANDWIDTH_IN), :, :, 1:].sum() +
                Cy[Resource.Y_resource_idx(
                    Resource.BANDWIDTH_IN), :, :, 1:, :].sum()
            ).item())
        self.server_cpu_usage.append(
            Cy[Resource.Y_resource_idx(Resource.CPU), :, :, 0, :].sum().item())
        self.server_memory_usage.append(
            Cy[Resource.Y_resource_idx(Resource.MEMORY), :, :, 0, :].sum().item())

        # Client resources

        def client_bandwidth_in_usage():
            return (Cx[Resource.X_resource_idx(Resource.BANDWIDTH_IN), :, :, 1:].sum(dim=(0, 1)) +
                    Cy[Resource.Y_resource_idx(Resource.BANDWIDTH_IN), :, :, 1:].sum(dim=(0, 1, 3)))

        def client_bandwidth_out_usage():
            return Cy[Resource.Y_resource_idx(Resource.BANDWIDTH_OUT), :, :, 1:].sum(dim=(0, 1, 3))

        def client_cpu_usage():
            return (Cx[Resource.X_resource_idx(Resource.CPU), :, :, 1:].sum(dim=(0, 1)) +
                    Cy[Resource.Y_resource_idx(Resource.CPU), :, :, 1:].sum(dim=(0, 1, 3)))

        def client_memory_usage():
            return (Cx[Resource.X_resource_idx(Resource.MEMORY), :, :, 1:].sum(dim=(0, 1)) +
                    Cy[Resource.Y_resource_idx(Resource.MEMORY), :, :, 1:].sum(dim=(0, 1, 3)))

        def client_storage_usage():
            return Cx[Resource.X_resource_idx(Resource.STORAGE), :, :, 1:].sum(dim=(0, 1))

        self.online_client_total_bandwidth_in_usage.append(
            client_bandwidth_in_usage()[O == 1].sum().item())
        self.online_client_total_bandwidth_out_usage.append(
            client_bandwidth_out_usage()[O == 1].sum().item())
        self.online_client_total_cpu_usage.append(
            client_cpu_usage()[O == 1].sum().item())
        self.online_client_total_memory_usage.append(
            client_memory_usage()[O == 1].sum().item())
        self.online_client_total_storage_usage.append(
            client_storage_usage()[O == 1].sum().item())

        self.online_client_mean_bandwidth_in_usage.append(
            client_bandwidth_in_usage()[O == 1].mean().item())
        self.online_client_mean_bandwidth_out_usage.append(
            client_bandwidth_out_usage()[O == 1].mean().item())
        self.online_client_mean_cpu_usage.append(
            client_cpu_usage()[O == 1].mean().item())
        self.online_client_mean_memory_usage.append(
            client_memory_usage()[O == 1].mean().item())
        self.online_client_mean_storage_usage.append(
            client_storage_usage()[O == 1].mean().item())

        self.online_client_std_bandwidth_in_usage.append(
            _safe_std(client_bandwidth_in_usage()[O == 1]))
        self.online_client_std_bandwidth_out_usage.append(
            _safe_std(client_bandwidth_out_usage()[O == 1]))
        self.online_client_std_cpu_usage.append(
            _safe_std(client_cpu_usage()[O == 1]))
        self.online_client_std_memory_usage.append(
            _safe_std(client_memory_usage()[O == 1]))
        self.online_client_std_storage_usage.append(
            _safe_std(client_storage_usage()[O == 1]))

        self.offline_client_total_storage_usage.append(
            client_storage_usage()[O == 0].sum().item())
        self.offline_client_mean_storage_usage.append(
            client_storage_usage()[O == 0].mean().item())
        self.offline_client_std_storage_usage.append(
            _safe_std(client_storage_usage()[O == 0]))

        # Subtasks

        subtasks_per_request = self._model_context.partition_mask.sum(
            dim=0).tolist()

        for t in range(self._n_tasks):
            q = t + 1
            n_subtasks = subtasks_per_request[t]

            self.required_total_y_subtasks[t].append(
                (Q == q).sum().item() * n_subtasks)

            self.server_total_executing_y_subtasks[t].append(
                (Ys[:, t, 0] == YState.EXECUTING).sum().item())
            self.server_total_done_y_subtasks[t].append(
                (Ys[:, t, 0] == YState.DONE).sum().item())

            self.online_client_total_downloading_x_subtasks[t].append(
                (X[:, t, 1:] == XState.DOWNLOADING)[:, O == 1].sum().item())
            self.online_client_total_paused_x_subtasks[t].append(
                (X[:, t, 1:] == XState.PAUSED)[:, O == 1].sum().item())
            self.online_client_total_ready_x_subtasks[t].append(
                (X[:, t, 1:] == XState.READY)[:, O == 1].sum().item())
            self.online_client_total_downloading_y_subtasks[t].append(
                (Ys[:, t, 1:] == YState.DOWNLOADING)[:, O == 1].sum().item())
            self.online_client_total_executing_y_subtasks[t].append(
                (Ys[:, t, 1:] == YState.EXECUTING)[:, O == 1].sum().item())
            self.online_client_total_uploading_y_subtasks[t].append(
                (Ys[:, t, 1:] == YState.UPLOADING)[:, O == 1].sum().item())
            self.online_client_total_done_y_subtasks[t].append(
                (Ys[:, t, 1:] == YState.DONE)[:, O == 1].sum().item())

            self.online_client_mean_downloading_x_subtasks[t].append(
                (X[:, t, 1:] == XState.DOWNLOADING).sum(dim=0)[O == 1].mean(dtype=torch.float32).item())
            self.online_client_mean_paused_x_subtasks[t].append(
                (X[:, t, 1:] == XState.PAUSED).sum(dim=0)[O == 1].mean(dtype=torch.float32).item())
            self.online_client_mean_ready_x_subtasks[t].append(
                (X[:, t, 1:] == XState.READY).sum(dim=0)[O == 1].mean(dtype=torch.float32).item())
            self.online_client_mean_downloading_y_subtasks[t].append(
                (Ys[:, t, 1:] == YState.DOWNLOADING).sum(dim=(0, 2))[O == 1].mean(dtype=torch.float32).item())
            self.online_client_mean_executing_y_subtasks[t].append(
                (Ys[:, t, 1:] == YState.EXECUTING).sum(dim=(0, 2))[O == 1].mean(dtype=torch.float32).item())
            self.online_client_mean_uploading_y_subtasks[t].append(
                (Ys[:, t, 1:] == YState.UPLOADING).sum(dim=(0, 2))[O == 1].mean(dtype=torch.float32).item())
            self.online_client_mean_done_y_subtasks[t].append(
                (Ys[:, t, 1:] == YState.DONE).sum(dim=(0, 2))[O == 1].mean(dtype=torch.float32).item())

            self.online_client_std_downloading_x_subtasks[t].append(
                _safe_std((X[:, t, 1:] == XState.DOWNLOADING).sum(dim=0)[O == 1]))
            self.online_client_std_paused_x_subtasks[t].append(
                _safe_std((X[:, t, 1:] == XState.PAUSED).sum(dim=0)[O == 1]))
            self.online_client_std_ready_x_subtasks[t].append(
                _safe_std((X[:, t, 1:] == XState.READY).sum(dim=0)[O == 1]))
            self.online_client_std_downloading_y_subtasks[t].append(
                _safe_std((Ys[:, t, 1:] == YState.DOWNLOADING).sum(dim=(0, 2))[O == 1]))
            self.online_client_std_executing_y_subtasks[t].append(
                _safe_std((Ys[:, t, 1:] == YState.EXECUTING).sum(dim=(0, 2))[O == 1]))
            self.online_client_std_uploading_y_subtasks[t].append(
                _safe_std((Ys[:, t, 1:] == YState.UPLOADING).sum(dim=(0, 2))[O == 1]))
            self.online_client_std_done_y_subtasks[t].append(
                _safe_std((Ys[:, t, 1:] == YState.DONE).sum(dim=(0, 2))[O == 1]))

            self.offline_client_total_downloading_x_subtasks[t].append(
                (X[:, t, 1:] == XState.DOWNLOADING)[:, O == 0].sum().item())
            self.offline_client_mean_downloading_x_subtasks[t].append(
                (X[:, t, 1:] == XState.DOWNLOADING).sum(dim=0)[O == 0].mean(dtype=torch.float32).item())
            self.offline_client_std_downloading_x_subtasks[t].append(
                _safe_std((X[:, t, 1:] == XState.DOWNLOADING).sum(dim=0)[O == 0]))
            self.offline_client_total_paused_x_subtasks[t].append(
                (X[:, t, 1:] == XState.PAUSED)[:, O == 0].sum().item())
            self.offline_client_mean_paused_x_subtasks[t].append(
                (X[:, t, 1:] == XState.PAUSED).sum(dim=0)[O == 0].mean(dtype=torch.float32).item())
            self.offline_client_std_paused_x_subtasks[t].append(
                _safe_std((X[:, t, 1:] == XState.PAUSED).sum(dim=0)[O == 0]))
            self.offline_client_total_ready_x_subtasks[t].append(
                (X[:, t, 1:] == XState.READY)[:, O == 0].sum().item())
            self.offline_client_mean_ready_x_subtasks[t].append(
                (X[:, t, 1:] == XState.READY).sum(dim=0)[O == 0].mean(dtype=torch.float32).item())
            self.offline_client_std_ready_x_subtasks[t].append(
                _safe_std((X[:, t, 1:] == XState.READY).sum(dim=0)[O == 0]))

        self.online_client_std_downloading_x_subtasks_all.append(
            _safe_std((X[:, :, 1:] == XState.DOWNLOADING).sum(dim=(0, 1))[O == 1]))
        self.online_client_std_paused_x_subtasks_all.append(
            _safe_std((X[:, :, 1:] == XState.PAUSED).sum(dim=(0, 1))[O == 1]))
        self.online_client_std_ready_x_subtasks_all.append(
            _safe_std((X[:, :, 1:] == XState.READY).sum(dim=(0, 1))[O == 1]))
        self.online_client_std_downloading_y_subtasks_all.append(
            _safe_std((Ys[:, :, 1:] == YState.DOWNLOADING).sum(dim=(0, 1, 3))[O == 1]))
        self.online_client_std_executing_y_subtasks_all.append(
            _safe_std((Ys[:, :, 1:] == YState.EXECUTING).sum(dim=(0, 1, 3))[O == 1]))
        self.online_client_std_uploading_y_subtasks_all.append(
            _safe_std((Ys[:, :, 1:] == YState.UPLOADING).sum(dim=(0, 1, 3))[O == 1]))
        self.online_client_std_done_y_subtasks_all.append(
            _safe_std((Ys[:, :, 1:] == YState.DONE).sum(dim=(0, 1, 3))[O == 1]))
        self.offline_client_std_downloading_x_subtasks_all.append(
            _safe_std((X[:, :, 1:] == XState.DOWNLOADING).sum(dim=(0, 1))[O == 0]))
        self.offline_client_std_ready_x_subtasks_all.append(
            _safe_std((X[:, :, 1:] == XState.READY).sum(dim=(0, 1))[O == 0]))
        self.offline_client_std_paused_x_subtasks_all.append(
            _safe_std((X[:, :, 1:] == XState.PAUSED).sum(dim=(0, 1))[O == 0]))

        # Subtask completion and aborts

        for t in range(self._n_tasks):
            q = t + 1

            server_y_subtasks_completed = ((Yshat[:, t, 0] != YState.DONE) & (
                Ys[:, t, 0] == YState.DONE)).sum().item()
            server_y_subtasks_aborted = (
                client_Ys_action[:, t, 0] == ClientYAction.ABORT).sum().item()

            client_x_subtasks_completed = ((Xhat[:, t, 1:] != XState.READY) & (
                X[:, t, 1:] == XState.READY) & (client_X_action[:, t, 1:] != ClientXAction.ABORT)).sum().item()
            client_x_subtasks_removed = ((X[:, t, 1:] == XState.READY) & (
                client_X_action[:, t, 1:] == ClientXAction.ABORT)).sum().item()
            client_x_subtasks_aborted = ((X[:, t, 1:] != XState.READY) & (
                client_X_action[:, t, 1:] == ClientXAction.ABORT)).sum().item()

            client_y_subtasks_completed = ((Yshat[:, t, 1:] != YState.DONE) & (
                Ys[:, t, 1:] == YState.DONE)).sum().item()
            client_y_subtasks_aborted = (
                client_Ys_action[:, t, 1:] == ClientYAction.ABORT).sum().item()

            if self._timestep > 0:
                self.cum_server_y_subtasks_completed[t].append(
                    self.cum_server_y_subtasks_completed[t][-1] + server_y_subtasks_completed)
                self.cum_server_y_subtasks_aborted[t].append(
                    self.cum_server_y_subtasks_aborted[t][-1] + server_y_subtasks_aborted)

                self.cum_client_x_subtasks_completed[t].append(
                    self.cum_client_x_subtasks_completed[t][-1] + client_x_subtasks_completed)
                self.cum_client_x_subtasks_removed[t].append(
                    self.cum_client_x_subtasks_removed[t][-1] + client_x_subtasks_removed)
                self.cum_client_x_subtasks_aborted[t].append(
                    self.cum_client_x_subtasks_aborted[t][-1] + client_x_subtasks_aborted)

                self.cum_client_y_subtasks_completed[t].append(
                    self.cum_client_y_subtasks_completed[t][-1] + client_y_subtasks_completed)
                self.cum_client_y_subtasks_aborted[t].append(
                    self.cum_client_y_subtasks_aborted[t][-1] + client_y_subtasks_aborted)
            else:
                self.cum_server_y_subtasks_completed[t].append(
                    server_y_subtasks_completed)
                self.cum_server_y_subtasks_aborted[t].append(
                    server_y_subtasks_aborted)

                self.cum_client_x_subtasks_completed[t].append(
                    client_x_subtasks_completed)
                self.cum_client_x_subtasks_removed[t].append(
                    client_x_subtasks_removed)
                self.cum_client_x_subtasks_aborted[t].append(
                    client_x_subtasks_aborted)

                self.cum_client_y_subtasks_completed[t].append(
                    client_y_subtasks_completed)
                self.cum_client_y_subtasks_aborted[t].append(
                    client_y_subtasks_aborted)

        A_used = torch.zeros_like(A)
        A_used[Resource.X_resources()] += Cx.sum(dim=(1, 2))
        A_used[Resource.Y_resources()] += Cy.sum(dim=(1, 2, 4))

        server_constraints_violated = (A_used[:, 0] > A[:, 0]).any().item()
        n_client_constraints_violations = (
            A_used[:, 1:] > A[:, 1:]).any(dim=0)[O == 1].sum().item()

        self.server_constraints_violation.append(server_constraints_violated)
        self.n_client_constraints_violations.append(
            n_client_constraints_violations)

        # Update vars

        self._timestep += 1
        self._prev_Q = Q
        self._prev_Tq = Tq
