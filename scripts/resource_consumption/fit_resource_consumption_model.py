import os
import argparse
import pandas as pd
from enum import Enum
from utils.resource_consumption_model import *
from utils.datetime import now_str


from models.experiments import ResourceConsumptionModelExperiment


def run(experiment):
    df_path = experiment.df_path
    series_name = experiment.series_name
    task_type = experiment.task_type
    task_state = experiment.task_state
    arima_order = tuple(experiment.arima_order)
    arima_trend = experiment.arima_trend
    output_dir = experiment.output_dir

    df_all = pd.read_csv(df_path)
    series_groups = {
        _normalize_conf(conf, task_state): [df[series_name].to_numpy()
                                            for _, df in df_trials.groupby("conf_trial")]
        for conf, df_trials in df_all.groupby(_get_conf_columns(task_state))
    }

    trainer = ResourceConsumptionModelTrainer(
        series_groups=series_groups,
        order=arima_order,
        trend=arima_trend,
    )
    config = trainer.fit()

    order_str = '-'.join(str(o) for o in arima_order)
    save_path = os.path.join(
        output_dir, f"consumption_{task_type.value}_{task_state.value}_{series_name}_model_{order_str}_{now_str()}.json")

    config.save(save_path, pretty=True)


def _get_conf_columns(task_state: ResourceConsumptionModelExperiment.TaskState):
    match task_state:
        case ResourceConsumptionModelExperiment.TaskState.DOWNLOAD:
            return ["conf_bandwidth_in"]
        case ResourceConsumptionModelExperiment.TaskState.EXECUTION:
            return ["conf_task_count", "conf_model_name"]
        case ResourceConsumptionModelExperiment.TaskState.UPLOAD:
            return ["conf_bandwidth_out"]


def _normalize_conf(conf, task_state: ResourceConsumptionModelExperiment.TaskState):
    match task_state:
        case ResourceConsumptionModelExperiment.TaskState.DOWNLOAD:
            return (int(conf[0]),)
        case ResourceConsumptionModelExperiment.TaskState.EXECUTION:
            return (int(conf[0]), str(conf[1]))
        case ResourceConsumptionModelExperiment.TaskState.UPLOAD:
            return (int(conf[0]),)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment handle")
    args = parser.parse_args()
    experiment = ResourceConsumptionModelExperiment(args.experiment)
    run(experiment)
