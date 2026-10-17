import torch
import pyro
import sys
from time import time

from simulation.model_callback import ModelCallback
from simulation.model_context import ModelContext
from simulation.types import *
from simulation.initial_distributions import InitialDistributions
from simulation.trajectory import Trajectory
from simulation.transition_functions import TransitionFunctions
from simulation.policies import Policies
import simulation.variable_names as var
from utils.datetime import format_time


class Model:
    def __init__(
        self,
        model_context: ModelContext,
        initial_distributions: InitialDistributions,
        transition_functions: TransitionFunctions,
        policies: Policies,
        trajectory: Trajectory,
        model_callback: ModelCallback = None,
    ):
        self.model_context = model_context
        self.initial_distributions = initial_distributions
        self.transition_functions = transition_functions
        self.policies = policies
        self.model_callback = model_callback
        self.trajectory = trajectory

    def run(self):
        A_dim = len(Resource.available_resources())
        Cx_dim = len(Resource.X_resources())
        Cy_dim = len(Resource.Y_resources())
        # No weights in this implementation
        W = torch.zeros((self.model_context.n_clients,))

        def append_vars():
            self.trajectory.append(O, To, Q, Tq, A, Xhat, Cx, Tx, X, Yihat, Yshat, Cy, Ty, Yi,
                                   Ys, W, server_X_action, client_X_action, server_Yi_action, client_Yi_action, server_Ys_action, client_Ys_action)

        def client_plate():
            return pyro.plate("i", self.model_context.n_clients)

        def node_plate():
            return pyro.plate("v", self.model_context.n_nodes)

        def task_plate():
            return pyro.plate("k", self.model_context.n_tasks)

        def partition_plate():
            return pyro.plate("p", self.model_context.n_partitions)

        def available_resource_plate():
            return pyro.plate("d", A_dim)

        def x_resource_plate():
            return pyro.plate("d", Cx_dim)

        def y_resource_plate():
            return pyro.plate("d", Cy_dim)

        def slot_plate():
            return pyro.plate("s", self.model_context.n_task_exec_slots)

        with client_plate():
            O = pyro.sample(var.O(t=0), self.initial_distributions.O)
            To = pyro.sample(var.To(t=0), self.initial_distributions.To)
            Q = pyro.sample(var.Q(t=0), self.initial_distributions.Q)
            Tq = pyro.sample(var.Tq(t=0), self.initial_distributions.Tq)

        with node_plate():
            with available_resource_plate():
                A = pyro.sample(var.A(t=0), self.initial_distributions.A)

            with task_plate(), partition_plate():
                Xhat = pyro.sample(
                    var.Xhat(t=0), self.initial_distributions.Xhat)
                with x_resource_plate():
                    Cx = pyro.sample(
                        var.Cx(t=0), self.initial_distributions.Cx)
                Tx = pyro.sample(
                    var.Tx(t=0), self.initial_distributions.Tx)
                X = pyro.sample(
                    var.X(t=0), self.initial_distributions.X)

        with slot_plate(), node_plate(), task_plate(), partition_plate():
            Yihat = pyro.sample(
                var.Yihat(t=0), self.initial_distributions.Yihat)
            Yshat = pyro.sample(
                var.Yshat(t=0), self.initial_distributions.Yshat)
            with y_resource_plate():
                Cy = pyro.sample(var.Cy(t=0), self.initial_distributions.Cy)
            Ty = pyro.sample(var.Ty(t=0), self.initial_distributions.Ty)
            Yi = pyro.sample(var.Yi(t=0), self.initial_distributions.Yi)
            Ys = pyro.sample(var.Ys(t=0), self.initial_distributions.Ys)

        server_X_action_dist, server_Yi_action_dist, server_Ys_action_dist = self.policies.client(
            O, To, Q, Tq, A, X, Tx, Cx, Yi, Ys, Ty, Cy, W)
        client_X_action_dist, client_Yi_action_dist, client_Ys_action_dist = self.policies.server(
            O, To, Q, Tq, A, X, Tx, Cx, Yi, Ys, Ty, Cy, W)

        with node_plate(), task_plate(), partition_plate():
            server_X_action = pyro.sample(
                var.server_X_action(t=0), server_X_action_dist)
            client_X_action = pyro.sample(
                var.client_X_action(t=0), client_X_action_dist)

        with slot_plate(), node_plate(), task_plate(), partition_plate():
            server_Yi_action = pyro.sample(
                var.server_Yi_action(t=0), server_Yi_action_dist)
            server_Ys_action = pyro.sample(
                var.server_Yi_action(t=0), server_Ys_action_dist)
            client_Yi_action = pyro.sample(
                var.client_Yi_action(t=0), client_Yi_action_dist)
            client_Ys_action = pyro.sample(
                var.client_Ys_action(t=0), client_Ys_action_dist)

        append_vars()

        start_time = time()
        stdout_is_tty = sys.stdout.isatty()

        for t in pyro.markov(range(1, self.model_context.n_timesteps + 1)):
            if self.model_callback:
                if not self.model_callback.on_step_begin(self, t):
                    break

            elapsed = time() - start_time
            eta = elapsed / t * (self.model_context.n_timesteps - t)
            progress = (
                f"Simulating time step {t + 1}/{self.model_context.n_timesteps} "
                f"| Time elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}"
            )

            if stdout_is_tty:
                print(progress, end="\r", flush=True)
            elif t == 1 or t % 100 == 0 or t == self.model_context.n_timesteps:
                print(progress, flush=True)

            with client_plate():
                To = pyro.sample(
                    var.To(t=t), self.transition_functions.To(To, O))
                O = pyro.sample(var.O(t=t), self.transition_functions.O(O, To))

            with node_plate(), task_plate(), partition_plate():
                Xhat = pyro.sample(var.Xhat(t=t), self.transition_functions.Xhat(
                    X, O, server_X_action, client_X_action))

            with slot_plate(), node_plate(), task_plate(), partition_plate():
                Yihat = pyro.sample(var.Yihat(t=t), self.transition_functions.Yihat(
                    Yi, Ys, O, server_Yi_action, client_Yi_action))
                Yshat = pyro.sample(var.Yshat(t=t), self.transition_functions.Yshat(
                    Ys, Yi, O, server_Ys_action, client_Ys_action))

            with client_plate():
                Tq = pyro.sample(
                    var.Tq(t=t), self.transition_functions.Tq(Tq, Q, O, Yi, Ys))
                Q = pyro.sample(var.Q(t=t), self.transition_functions.Q(
                    Q, O, Tq))

            with node_plate():
                with available_resource_plate():
                    A = pyro.sample(
                        var.A(t=t), self.transition_functions.A(A, O))

            Cx_dist, Cy_dist = self.transition_functions.C(
                Cx, Cy, Xhat, Yihat, Yshat, Tx, Ty, A)

            with node_plate(), task_plate(), partition_plate():
                with x_resource_plate():
                    Cx = pyro.sample(var.Cx(t=t), Cx_dist)
                Tx = pyro.sample(
                    var.Tx(t=t), self.transition_functions.Tx(Tx, Xhat, Cx))
                X = pyro.sample(
                    var.X(t=t), self.transition_functions.X(Xhat, Tx))

            with slot_plate(), node_plate(), task_plate(), partition_plate():
                with y_resource_plate():
                    Cy = pyro.sample(var.Cy(t=t), Cy_dist)
                Ty = pyro.sample(
                    var.Ty(t=t), self.transition_functions.Ty(Ty, Yihat, Yshat, Cy))
                Yi = pyro.sample(
                    var.Yi(t=t), self.transition_functions.Yi(Yihat, Yshat, Ty))
                Ys = pyro.sample(
                    var.Ys(t=t), self.transition_functions.Ys(Yshat, Yihat, Ty))

            server_X_action_dist, server_Yi_action_dist, server_Ys_action_dist = self.policies.server(
                O, To, Q, Tq, A, X, Tx, Cx, Yi, Ys, Ty, Cy, W)
            client_X_action_dist, client_Yi_action_dist, client_Ys_action_dist = self.policies.client(
                O, To, Q, Tq, A, X, Tx, Cx, Yi, Ys, Ty, Cy, W)

            with node_plate(), task_plate(), partition_plate():
                server_X_action = pyro.sample(
                    var.server_X_action(t=0), server_X_action_dist)
                client_X_action = pyro.sample(
                    var.client_X_action(t=0), client_X_action_dist)

            with slot_plate(), node_plate(), task_plate(), partition_plate():
                server_Yi_action = pyro.sample(
                    var.server_Yi_action(t=0), server_Yi_action_dist)
                server_Ys_action = pyro.sample(
                    var.server_Ys_action(t=0), server_Ys_action_dist)
                client_Yi_action = pyro.sample(
                    var.client_Yi_action(t=0), client_Yi_action_dist)
                client_Ys_action = pyro.sample(
                    var.client_Ys_action(t=0), client_Ys_action_dist)

            append_vars()

            if self.model_callback:
                if not self.model_callback.on_step_end(self, t):
                    break

        if stdout_is_tty:
            print(flush=True)
