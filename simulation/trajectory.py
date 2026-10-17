import torch
from abc import ABC, abstractmethod


class Trajectory(ABC):
    @abstractmethod
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
        pass
