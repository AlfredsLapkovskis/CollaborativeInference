def O(i=None, t=None):
    return _make_var_with_index("O", i=i, t=t)


def To(i=None, t=None):
    return _make_var_with_index("To", i=i, t=t)


def Q(i=None, t=None):
    return _make_var_with_index("Q", i=i, t=t)


def Tq(i=None, t=None):
    return _make_var_with_index("Tq", i=i, t=t)


def A(v=None, t=None, d=None):
    return _make_var_with_index("A", v=v, t=t, d=d)


def Xhat(v=None, t=None, k=None, p=None):
    return _make_var_with_index("Xhat", v=v, t=t, k=k, p=p)


def Cx(v=None, t=None, d=None, k=None, p=None):
    return _make_var_with_index("Cx", v=v, t=t, d=d, k=k, p=p)


def Tx(v=None, t=None, k=None, p=None):
    return _make_var_with_index("Tx", v=v, t=t, k=k, p=p)


def X(v=None, t=None, k=None, p=None):
    return _make_var_with_index("X", v=v, t=t, k=k, p=p)


def Yihat(v=None, t=None, k=None, p=None, s=None):
    return _make_var_with_index("Yihat", v=v, t=t, k=k, p=p, s=s)


def Yshat(v=None, t=None, k=None, p=None, s=None):
    return _make_var_with_index("Yshat", v=v, t=t, k=k, p=p, s=s)


def Cy(v=None, t=None, d=None, k=None, p=None, s=None):
    return _make_var_with_index("Cy", v=v, t=t, d=d, k=k, p=p, s=s)


def Ty(v=None, t=None, k=None, p=None, s=None):
    return _make_var_with_index("Ty", v=v, t=t, k=k, p=p, s=s)


def Yi(v=None, t=None, k=None, p=None, s=None):
    return _make_var_with_index("Yi", v=v, t=t, k=k, p=p, s=s)


def Ys(v=None, t=None, k=None, p=None, s=None):
    return _make_var_with_index("Ys", v=v, t=t, k=k, p=p, s=s)


def server_X_action(v=None, t=None, k=None, p=None):
    return _make_var_with_index("server_X_action", v=v, t=t, k=k, p=p)


def client_X_action(i=None, t=None, k=None, p=None):
    return _make_var_with_index("client_X_action", i=i, t=t, k=k, p=p)


def server_Yi_action(v=None, t=None, k=None, p=None, s=None):
    return _make_var_with_index("server_Yi_action", v=v, t=t, k=k, p=p, s=s)


def server_Ys_action(v=None, t=None, k=None, p=None, s=None):
    return _make_var_with_index("server_Ys_action", v=v, t=t, k=k, p=p, s=s)


def client_Yi_action(i=None, t=None, k=None, p=None, s=None):
    return _make_var_with_index("client_Yi_action", i=i, t=t, k=k, p=p, s=s)


def client_Ys_action(i=None, t=None, k=None, p=None, s=None):
    return _make_var_with_index("client_Ys_action", i=i, t=t, k=k, p=p, s=s)


def _make_var_with_index(var, **kwargs):
    index = "_".join({f"{k}={v}" for k, v in kwargs.items() if v is not None})
    return f"{var}_{index}" if index else var
