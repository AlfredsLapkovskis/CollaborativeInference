import torch


def get_device(device: str) -> torch.device:
    match device:
        case "cpu":
            return torch.device("cpu")
        case "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                raise ValueError("CUDA device is not available.")
        case "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                raise ValueError("MPS device is not available.")
        case "gpu":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                raise ValueError("No GPU device is available.")
        case "default" | "" | "auto" | None:
            return torch.device("cpu")
        case _:
            raise ValueError(f"Invalid device: {device}")
