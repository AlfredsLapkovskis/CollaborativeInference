import torch
from abc import ABC, abstractmethod
from torch.distributions import Distribution


class Policies:
    @property
    def server(self):
        return self._server

    @property
    def client(self):
        return self._client

    def __init__(self, server: "ServerPolicy", client: "ClientPolicy"):
        self._server = server
        self._client = client


class _Policy(ABC):
    @abstractmethod
    def __call__(
        self,
        O: torch.Tensor,
        To: torch.Tensor,
        Q: torch.Tensor,
        Tq: torch.Tensor,
        A: torch.Tensor,
        X: torch.Tensor,
        Tx: torch.Tensor,
        Cx: torch.Tensor,
        Yi: torch.Tensor,
        Ys: torch.Tensor,
        Ty: torch.Tensor,
        Cy: torch.Tensor,
        W: torch.Tensor
    ) -> tuple[Distribution, Distribution, Distribution]:
        """
        Returns distributions over actions X, Yi, Ys.
        """
        pass


class ServerPolicy(_Policy):
    pass


class ClientPolicy(_Policy):
    pass
