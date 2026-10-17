import torch
from torch.distributions import Distribution
import pyro.distributions as dist

from simulation.types import *
from simulation.model_context import ModelContext


class InitialDistributions:
    @property
    def O(self):
        return self._O

    @property
    def To(self):
        return self._To

    @property
    def Q(self):
        return self._Q

    @property
    def Tq(self):
        return self._Tq

    @property
    def A(self):
        return self._A

    @property
    def Xhat(self):
        return self._Xhat

    @property
    def Cx(self):
        return self._Cx

    @property
    def Tx(self):
        return self._Tx

    @property
    def X(self):
        return self._X

    @property
    def Yihat(self):
        return self._Yihat

    @property
    def Yshat(self):
        return self._Yshat

    @property
    def Cy(self):
        return self._Cy

    @property
    def Ty(self):
        return self._Ty

    @property
    def Yi(self):
        return self._Yi

    @property
    def Ys(self):
        return self._Ys

    @property
    def W(self):
        return self._W

    def __init__(
        self,
        context: ModelContext,
        O: Distribution = None,
        To: Distribution = None,
        Q: Distribution = None,
        Tq: Distribution = None,
        A: Distribution = None,
        Xhat: Distribution = None,
        Cx: Distribution = None,
        Tx: Distribution = None,
        X: Distribution = None,
        Yihat: Distribution = None,
        Yshat: Distribution = None,
        Cy: Distribution = None,
        Ty: Distribution = None,
        Yi: Distribution = None,
        Ys: Distribution = None,
        W: Distribution = None,
    ):
        def X_default():
            n_nodes = context.n_nodes

            X = torch.full((context.n_partitions, context.n_tasks,
                           n_nodes), fill_value=XState.EMPTY, dtype=torch.int8)

            # server node always has all partitions
            X[:, :, 0][context.partition_mask] = XState.READY

            return dist.Delta(X)

        float_type = context.float_type

        self._O = O or dist.Delta(torch.tensor([0], dtype=torch.int8))
        self._To = To or dist.Delta(torch.tensor([0], dtype=torch.int32))
        self._Q = Q or dist.Delta(torch.tensor([0], dtype=torch.int32))
        self._Tq = Tq or dist.Delta(torch.tensor([0], dtype=torch.int32))
        self._A = A or dist.Delta(torch.tensor([0], dtype=float_type))
        self._Xhat = Xhat or dist.Delta(torch.tensor([0], dtype=torch.int8))
        self._Cx = Cx or dist.Delta(torch.tensor([0], dtype=float_type))
        self._Tx = Tx or dist.Delta(torch.tensor([0], dtype=torch.int32))
        self._X = X or X_default()
        self._Yihat = Yihat or dist.Delta(
            torch.tensor([0], dtype=torch.int32))
        self._Yshat = Yshat or dist.Delta(
            torch.tensor([0], dtype=torch.int8))
        self._Cy = Cy or dist.Delta(torch.tensor([0], dtype=float_type))
        self._Ty = Ty or dist.Delta(torch.tensor([0], dtype=torch.int32))
        self._Yi = Yi or dist.Delta(torch.tensor([0], dtype=torch.int32))
        self._Ys = Ys or dist.Delta(torch.tensor([0], dtype=torch.int8))
        self._W = W or dist.Delta(torch.tensor([0], dtype=torch.int32))
