import os
import argparse
import pandas as pd

from utils.task_completion_model import *
from utils.datetime import now_str
from models.experiments import TaskCompletionModelExperiment


def run(experiment):
    df_path = experiment.df_path
    task_type = experiment.task_type
    output_dir = experiment.output_dir
    standardize = experiment.standardize
    polynomial_degree = experiment.polynomial_degree

    conf_columns = ["conf_trial", "conf_bandwidth_in",
                    "conf_bandwidth_out", "conf_task_count", "conf_model_name"]
    stratification_columns = ["conf_model_name"]
    constant_feature_columns = ["conf_task_count"]
    variable_feature_columns = _get_resource_columns(task_type)
    ignored_feature_columns = _get_ignored_features(task_type)

    df_all = pd.read_csv(df_path)

    trainer = TaskCompletionModelTrainer(
        data=df_all,
        conf_columns=conf_columns,
        stratification_columns=stratification_columns,
        constant_feature_columns=constant_feature_columns,
        variable_feature_columns=variable_feature_columns,
        ignored_feature_columns=ignored_feature_columns,
        standardize=standardize,
        polynomial_degree=polynomial_degree,
    )
    config = trainer.fit()

    save_path = os.path.join(
        output_dir, f"task_completion_{task_type.value}_model_{now_str()}.json")

    config.save(save_path, pretty=True)


def _get_resource_columns(task_type: TaskCompletionModelExperiment.TaskType):
    return ["res_net_read", "res_net_write", "res_cpu", "res_memory"]


def _get_ignored_features(task_type: TaskCompletionModelExperiment.TaskType):
    match task_type:
        case TaskCompletionModelExperiment.TaskType.DOWNLOAD:
            return ["res_net_write"]
        case TaskCompletionModelExperiment.TaskType.EXECUTION:
            return ["res_net_read", "res_net_write"]
        case TaskCompletionModelExperiment.TaskType.UPLOAD:
            return ["res_net_read"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment handle")
    args = parser.parse_args()
    experiment = TaskCompletionModelExperiment(args.experiment)
    run(experiment)
