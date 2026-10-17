import torch
from simulation.trajectory import Trajectory


class TrajectoryDefault(Trajectory):
    @property
    def O_trajectory(self):
        return self._O_trajectory

    @property
    def To_trajectory(self):
        return self._To_trajectory

    @property
    def Q_trajectory(self):
        return self._Q_trajectory

    @property
    def Tq_trajectory(self):
        return self._Tq_trajectory

    @property
    def A_trajectory(self):
        return self._A_trajectory

    @property
    def Xhat_trajectory(self):
        return self._Xhat_trajectory

    @property
    def Cx_trajectory(self):
        return self._Cx_trajectory

    @property
    def Tx_trajectory(self):
        return self._Tx_trajectory

    @property
    def X_trajectory(self):
        return self._X_trajectory

    @property
    def Yihat_trajectory(self):
        return self._Yihat_trajectory

    @property
    def Yshat_trajectory(self):
        return self._Yshat_trajectory

    @property
    def Cy_trajectory(self):
        return self._Cy_trajectory

    @property
    def Ty_trajectory(self):
        return self._Ty_trajectory

    @property
    def Yi_trajectory(self):
        return self._Yi_trajectory

    @property
    def Ys_trajectory(self):
        return self._Ys_trajectory

    @property
    def W_trajectory(self):
        return self._W_trajectory

    @property
    def server_X_action_trajectory(self):
        return self._server_X_action_trajectory

    @property
    def client_X_action_trajectory(self):
        return self._client_X_action_trajectory

    @property
    def server_Yi_action_trajectory(self):
        return self._server_Yi_action_trajectory

    @property
    def client_Yi_action_trajectory(self):
        return self._client_Yi_action_trajectory

    @property
    def server_Ys_action_trajectory(self):
        return self._server_Ys_action_trajectory

    @property
    def client_Ys_action_trajectory(self):
        return self._client_Ys_action_trajectory

    def __init__(self):
        self._O_trajectory = []
        self._To_trajectory = []
        self._Q_trajectory = []
        self._Tq_trajectory = []
        self._A_trajectory = []
        self._Xhat_trajectory = []
        self._Cx_trajectory = []
        self._Tx_trajectory = []
        self._X_trajectory = []
        self._Yihat_trajectory = []
        self._Yshat_trajectory = []
        self._Cy_trajectory = []
        self._Ty_trajectory = []
        self._Yi_trajectory = []
        self._Ys_trajectory = []
        self._W_trajectory = []
        self._server_X_action_trajectory = []
        self._client_X_action_trajectory = []
        self._server_Yi_action_trajectory = []
        self._client_Yi_action_trajectory = []
        self._server_Ys_action_trajectory = []
        self._client_Ys_action_trajectory = []

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
        self._O_trajectory.append(O)
        self._To_trajectory.append(To)
        self._Q_trajectory.append(Q)
        self._Tq_trajectory.append(Tq)
        self._A_trajectory.append(A)
        self._Xhat_trajectory.append(Xhat)
        self._Cx_trajectory.append(Cx)
        self._Tx_trajectory.append(Tx)
        self._X_trajectory.append(X)
        self._Yihat_trajectory.append(Yihat)
        self._Yshat_trajectory.append(Yshat)
        self._Cy_trajectory.append(Cy)
        self._Ty_trajectory.append(Ty)
        self._Yi_trajectory.append(Yi)
        self._Ys_trajectory.append(Ys)
        self._W_trajectory.append(W)
        self._server_X_action_trajectory.append(server_X_action)
        self._client_X_action_trajectory.append(client_X_action)
        self._server_Yi_action_trajectory.append(server_Yi_action)
        self._client_Yi_action_trajectory.append(client_Yi_action)
        self._server_Ys_action_trajectory.append(server_Ys_action)
        self._client_Ys_action_trajectory.append(client_Ys_action)
