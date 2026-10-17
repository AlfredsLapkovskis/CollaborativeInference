import torch
from torch.distributions import Distribution
from abc import ABC, abstractmethod


class TransitionFunctions:
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
    def C(self):
        return self._C

    @property
    def Xhat(self):
        return self._Xhat

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
        O: "OTransitionFunction",
        To: "ToTransitionFunction",
        Q: "QTransitionFunction",
        Tq: "TqTransitionFunction",
        A: "ATransitionFunction",
        C: "CTransitionFunction",
        Xhat: "XhatTransitionFunction",
        Tx: "TxTransitionFunction",
        X: "XTransitionFunction",
        Yihat: "YihatTransitionFunction",
        Yshat: "YshatTransitionFunction",
        Ty: "TyTransitionFunction",
        Yi: "YiTransitionFunction",
        Ys: "YsTransitionFunction",
        W: "WTransitionFunction",
    ):
        self._O = O
        self._To = To
        self._Q = Q
        self._Tq = Tq
        self._A = A
        self._C = C
        self._Xhat = Xhat
        self._Tx = Tx
        self._X = X
        self._Yihat = Yihat
        self._Yshat = Yshat
        self._Ty = Ty
        self._Yi = Yi
        self._Ys = Ys
        self._W = W


class OTransitionFunction(ABC):
    @abstractmethod
    def __call__(self, O_prev: torch.Tensor, To_curr: torch.Tensor) -> Distribution:
        pass


class ToTransitionFunction(ABC):
    @abstractmethod
    def __call__(self, To_prev: torch.Tensor, O_prev: torch.Tensor) -> Distribution:
        pass


class QTransitionFunction(ABC):
    @abstractmethod
    def __call__(self, Q_prev: torch.Tensor, O_curr: torch.Tensor, Tq_curr: torch.Tensor) -> Distribution:
        pass


class TqTransitionFunction(ABC):
    @abstractmethod
    def __call__(self, Tq_prev: torch.Tensor, Q_prev: torch.Tensor, O_curr: torch.Tensor, Yi_prev: torch.Tensor, Ys_prev: torch.Tensor) -> Distribution:
        pass


class ATransitionFunction(ABC):
    @abstractmethod
    def __call__(self, A_prev: torch.Tensor, O_curr: torch.Tensor) -> Distribution:
        pass


class XhatTransitionFunction(ABC):
    @abstractmethod
    def __call__(self, X_prev: torch.Tensor, O_curr: torch.Tensor, server_actions: torch.Tensor, client_actions: torch.Tensor) -> Distribution:
        pass


class TxTransitionFunction(ABC):
    @abstractmethod
    def __call__(self, Tx_prev: torch.Tensor, Xhat_curr: torch.Tensor, Cx_curr: torch.Tensor) -> Distribution:
        pass


class XTransitionFunction(ABC):
    @abstractmethod
    def __call__(self, Xhat_curr: torch.Tensor, Tx_curr: torch.Tensor) -> Distribution:
        pass


class YihatTransitionFunction(ABC):
    @abstractmethod
    def __call__(self, Yi_prev: torch.Tensor, Ys_prev: torch.Tensor, O_curr: torch.Tensor, server_actions: torch.Tensor, client_actions: torch.Tensor) -> Distribution:
        pass


class YshatTransitionFunction(ABC):
    @abstractmethod
    def __call__(self, Ys_prev: torch.Tensor, Yi_prev: torch.Tensor, O_curr: torch.Tensor, server_actions: torch.Tensor, client_actions: torch.Tensor) -> Distribution:
        pass


class TyTransitionFunction(ABC):
    @abstractmethod
    def __call__(self, Ty_prev: torch.Tensor, Yihat_curr: torch.Tensor, Yshat_curr: torch.Tensor, Cy_curr: torch.Tensor) -> Distribution:
        pass


class CTransitionFunction(ABC):
    @abstractmethod
    def __call__(self, Cx_prev: torch.Tensor, Cy_prev: torch.Tensor, Xhat_curr: torch.Tensor, Yihat_curr: torch.Tensor, Yshat_curr: torch.Tensor, Tx_prev: torch.Tensor, Ty_prev: torch.Tensor, A_curr: torch.Tensor) -> tuple[Distribution, Distribution]:
        """
        Returns:
            A tuple of (Cx_curr_dist, Cy_curr_dist)
        """
        pass


class YiTransitionFunction(ABC):
    @abstractmethod
    def __call__(self, Yihat_curr: torch.Tensor, Yshat_curr: torch.Tensor, Ty_curr: torch.Tensor) -> Distribution:
        pass


class YsTransitionFunction(ABC):
    @abstractmethod
    def __call__(self, Yshat_curr: torch.Tensor, Yihat_curr: torch.Tensor, Ty_curr: torch.Tensor) -> Distribution:
        pass


class WTransitionFunction(ABC):
    @abstractmethod
    def __call__(self, W_prev: torch.Tensor, Q_curr: torch.Tensor, Y_curr: torch.Tensor, Tq_curr: torch.Tensor, Ty_curr: torch.Tensor) -> Distribution:
        pass
