import argparse
import numpy as np
import pandas as pd

from utils.task_completion_model import *


# Number of augmented copies to generate per real subject.
# Each copy gets a different random pattern of throttling intervals.
_N_AUGMENTED_PER_SUBJECT = 4

# Range of throttle scaling factors. A factor of 0.5 means the CPU (and correlated
# resources) drop to 50% of their original value during the throttled interval,
# and that segment of the trajectory is stretched to 2x its original length.
_THROTTLE_SCALE_RANGE = (0.1, 0.9)

# Range for randomly sampling the episode size as a fraction of trajectory length.
# A fresh fraction is drawn for each augmented copy, giving structural diversity
# on top of per-episode throttle intensity variation.
_THROTTLE_EPISODE_LENGTH_RANGE = (0.1, 0.5)

# Probability that any given episode slot is actually throttled.
# Controls how dense throttling is within an augmented subject.
_THROTTLE_EPISODE_PROB = 0.5


def _stretch_segment(
    segment: pd.DataFrame,
    scale: float,
    resource_cols: list[str],
) -> pd.DataFrame:
    """
    Stretch a single trajectory segment by interpolating resource columns over
    a longer timeline (1/scale times as many rows) and scaling down their magnitudes.
    Timestamps are excluded here and assigned globally after stitching.
    Non-resource columns are forward-filled from the nearest original index.
    """
    n_orig = len(segment)
    n_stretched = max(1, round(n_orig / scale))

    orig_x = np.linspace(0, 1, n_orig)
    target_x = np.linspace(0, 1, n_stretched)

    stretched = {}
    for col in segment.columns:
        if col == "timestamp":
            continue  # assigned globally after all segments are stitched
        elif col in resource_cols:
            interpolated = np.interp(target_x, orig_x, segment[col].values)
            stretched[col] = interpolated * scale
        else:
            idx = np.clip(np.searchsorted(orig_x, target_x,
                          side="right") - 1, 0, n_orig - 1)
            stretched[col] = segment[col].values[idx]

    return pd.DataFrame(stretched)


def _apply_random_throttling(
    group: pd.DataFrame,
    scale_cols: list[str],
    scale_range: tuple[float, float],
    episode_length_range: tuple[float, float],
    episode_prob: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Split a subject's trajectory into fixed-size episode slots, then randomly
    throttle each slot independently. Throttled slots are stretched (more rows,
    lower resource values); normal slots are kept as-is. Timestamps are assigned
    last as a strict second-by-second sequence starting from the subject's first
    timestamp.

    The episode size is drawn fresh for each call, giving structural diversity
    across augmented copies of the same subject.
    """
    n = len(group)
    episode_size = max(1, round(n * rng.uniform(*episode_length_range)))

    splits = [group.iloc[i:i + episode_size]
              for i in range(0, n, episode_size)]

    processed = []
    for segment in splits:
        if rng.random() < episode_prob:
            scale = rng.uniform(*scale_range)
            processed.append(_stretch_segment(segment, scale, scale_cols))
        else:
            processed.append(segment.drop(columns="timestamp"))

    result = pd.concat(processed, ignore_index=True)

    # Assign strict second-by-second timestamps starting from the subject's first timestamp.
    t0 = pd.to_datetime(group["timestamp"].iloc[0])
    result["timestamp"] = [
        str(t0 + pd.Timedelta(seconds=i)) for i in range(len(result))
    ]

    return result


def _augment_throttled(
    task_state: str,
    df: pd.DataFrame,
    conf_columns: list[str],
    n_augmented: int = _N_AUGMENTED_PER_SUBJECT,
    scale_range: tuple[float, float] = _THROTTLE_SCALE_RANGE,
    episode_length_range: tuple[float, float] = _THROTTLE_EPISODE_LENGTH_RANGE,
    episode_prob: float = _THROTTLE_EPISODE_PROB,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    For each real subject, generate n_augmented synthetic copies where throttling
    episodes randomly appear and disappear throughout the trajectory. Each episode
    independently decides whether throttling occurs and at what intensity.

    This creates within-subject contrast — the model sees the same subject before,
    during, and after throttling — which directly teaches it the causal direction.
    """

    match task_state:
        case "download":
            scale_cols = ["res_cpu", "res_memory", "res_net_read"]
        case "execution":
            scale_cols = ["res_cpu"]
        case "upload":
            scale_cols = ["res_cpu", "res_memory", "res_net_write"]
        case _:
            raise ValueError(f"Unknown task_state: {task_state}")

    rng = np.random.default_rng(rng_seed)
    augmented_groups = []

    # Assign augmented subjects unique conf_trial values that don't collide with
    # any existing ones, so TaskCompletionModelTrainer groups them as separate subjects.
    next_trial_id = int(df["conf_trial"].max()) + 1

    n_changed = 0
    n_same = 0

    for _, group in df.groupby(conf_columns):
        for _ in range(n_augmented):
            aug = _apply_random_throttling(
                group, scale_cols, scale_range, episode_length_range, episode_prob, rng
            )
            aug["conf_trial"] = next_trial_id
            if len(aug) != len(group):
                n_changed += 1
            else:
                n_same += 1
            next_trial_id += 1
            augmented_groups.append(aug)

    df_aug = pd.concat(augmented_groups, ignore_index=True)
    return pd.concat([df, df_aug], ignore_index=True)


def run(args):
    src_path = args.src_path
    output_path = args.output_path
    task_state = args.task_state

    df = pd.read_csv(src_path)

    conf_columns = ["conf_trial", "conf_bandwidth_in",
                    "conf_bandwidth_out", "conf_task_count", "conf_model_name"]

    df_augmented = _augment_throttled(task_state, df, conf_columns)
    df_augmented.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, help="Path to input CSV file.")
    parser.add_argument("--output_path", type=str,
                        help="Path to output augmented CSV file.")
    parser.add_argument("--task_state", type=str, choices=["download", "execution", "upload"], 
                        help="Task state for which to augment data.")

    args = parser.parse_args()

    run(args)
