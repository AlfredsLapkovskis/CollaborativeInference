import pytest
import torch
from copy import deepcopy

from utils.vector_arima_simulator import VectorArimaSimulator
from utils.task_completion_model import TaskCompletionModel
from utils.vector_plr_simulator import VectorPLRSimulator


class TestVectorCoxPHSimulator:
    @pytest.fixture(
        params=[
            1000,
        ],
    )
    def n_series(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [
                TaskCompletionModel.deserialize("""
{"stratification_columns": ["conf_model_name"], "constant_feature_columns": ["conf_task_count"], "variable_feature_columns": ["res_net_read", "res_net_write", "res_cpu", "res_memory"], "ignored_feature_columns": ["res_net_write"], "standardize": true, "means": [0.6806772351264954, 16.42621612548828, 91263352.0, 0.0, 33.13872528076172, 89089264.0], "stds": [0.7593519687652588, 18.117351531982422, 130039280.0, 1.0, 34.30669403076172, 116214032.0], "polynomial_degree": 2, "polynomial_powers": [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], "model_strata": ["densenet169", "densenet201", "googlenet", "resnet152", "vgg13", "vgg16", "vgg19", "vit-b-16", "vit-b-32", "vit-h-14", "vit-l-16", "vit-l-32"], "model_stratum_dummy_names": ["_s_densenet169", "_s_densenet201", "_s_googlenet", "_s_resnet152", "_s_vgg13", "_s_vgg16", "_s_vgg19", "_s_vit-b-16", "_s_vit-b-32", "_s_vit-h-14", "_s_vit-l-16", "_s_vit-l-32"], "model_param_names": ["_s_densenet169", "_s_densenet201", "_s_googlenet", "_s_resnet152", "_s_vgg13", "_s_vgg16", "_s_vgg19", "_s_vit-b-16", "_s_vit-b-32", "_s_vit-h-14", "_s_vit-l-16", "_s_vit-l-32", "conf_task_count", "res_net_read", "res_net_write", "res_cpu", "res_memory", "log_t conf_task_count", "log_t res_net_read", "log_t res_net_write", "log_t res_cpu", "log_t res_memory", "conf_task_count^2", "conf_task_count res_net_read", "conf_task_count res_net_write", "conf_task_count res_cpu", "conf_task_count res_memory", "res_net_read^2", "res_net_read res_net_write", "res_net_read res_cpu", "res_net_read res_memory", "res_net_write^2", "res_net_write res_cpu", "res_net_write res_memory", "res_cpu^2", "res_cpu res_memory", "res_memory^2", "_s_densenet169 log_t", "_s_densenet169 log_t^2", "_s_densenet201 log_t", "_s_densenet201 log_t^2", "_s_googlenet log_t", "_s_googlenet log_t^2", "_s_resnet152 log_t", "_s_resnet152 log_t^2", "_s_vgg13 log_t", "_s_vgg13 log_t^2", "_s_vgg16 log_t", "_s_vgg16 log_t^2", "_s_vgg19 log_t", "_s_vgg19 log_t^2", "_s_vit-b-16 log_t", "_s_vit-b-16 log_t^2", "_s_vit-b-32 log_t", "_s_vit-b-32 log_t^2", "_s_vit-h-14 log_t", "_s_vit-h-14 log_t^2", "_s_vit-l-16 log_t", "_s_vit-l-16 log_t^2", "_s_vit-l-32 log_t", "_s_vit-l-32 log_t^2"], "model_params": [-0.779431164264679, -0.7862350344657898, -0.6112536191940308, -0.6393244862556458, -0.2626190781593323, -0.2758786976337433, -0.2617676556110382, -0.662828266620636, -0.13439126312732697, 0.18196411430835724, -0.0940527692437172, 0.04660242423415184, -0.5716272592544556, 0.8466007113456726, 0.0, 8.723784446716309, -5.675726890563965, -0.20534226298332214, 1.1398804187774658, 0.0, -1.377754807472229, 0.5490115880966187, 0.37488678097724915, -0.3546765148639679, 0.0, -0.15832525491714478, 0.21024899184703827, 0.34946709871292114, 0.0, 1.0926567316055298, -0.9296344518661499, 0.0, 0.0, 0.0, -1.169485330581665, -0.654019296169281, 1.421453833580017, 1.7786489725112915, -0.15308845043182373, 1.7788448333740234, -0.15805469453334808, 1.8602962493896484, -0.19283875823020935, 1.8601683378219604, -0.1849163919687271, 1.9573590755462646, -0.25579991936683655, 1.928048849105835, -0.24989525973796844, 1.9059573411941528, -0.24522441625595093, 1.7881301641464233, -0.16004236042499542, 1.9962995052337646, -0.29159438610076904, 2.016730308532715, -0.29996564984321594, 1.920695424079895, -0.2510523498058319, 2.009742259979248, -0.300809383392334], "model_aic": 301345.1453974041}
"""),
                TaskCompletionModel.deserialize("""
{"stratification_columns": ["conf_model_name"], "constant_feature_columns": ["conf_task_count"], "variable_feature_columns": ["res_net_read", "res_net_write", "res_cpu", "res_memory"], "ignored_feature_columns": ["res_net_read", "res_net_write"], "standardize": true, "means": [2.0223422050476074, 26.922685623168945, 0.0, 0.0, 157.90048217773438, 1455227904.0], "stds": [1.415396809577942, 20.206939697265625, 1.0, 1.0, 149.7842254638672, 1691732352.0], "polynomial_degree": 2, "polynomial_powers": [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], "model_strata": ["densenet169", "densenet201", "googlenet", "resnet152", "vgg13", "vgg16", "vgg19", "vit-b-16", "vit-b-32", "vit-h-14", "vit-l-16", "vit-l-32"], "model_stratum_dummy_names": ["_s_densenet169", "_s_densenet201", "_s_googlenet", "_s_resnet152", "_s_vgg13", "_s_vgg16", "_s_vgg19", "_s_vit-b-16", "_s_vit-b-32", "_s_vit-h-14", "_s_vit-l-16", "_s_vit-l-32"], "model_param_names": ["_s_densenet169", "_s_densenet201", "_s_googlenet", "_s_resnet152", "_s_vgg13", "_s_vgg16", "_s_vgg19", "_s_vit-b-16", "_s_vit-b-32", "_s_vit-h-14", "_s_vit-l-16", "_s_vit-l-32", "conf_task_count", "res_net_read", "res_net_write", "res_cpu", "res_memory", "log_t conf_task_count", "log_t res_net_read", "log_t res_net_write", "log_t res_cpu", "log_t res_memory", "conf_task_count^2", "conf_task_count res_net_read", "conf_task_count res_net_write", "conf_task_count res_cpu", "conf_task_count res_memory", "res_net_read^2", "res_net_read res_net_write", "res_net_read res_cpu", "res_net_read res_memory", "res_net_write^2", "res_net_write res_cpu", "res_net_write res_memory", "res_cpu^2", "res_cpu res_memory", "res_memory^2", "_s_densenet169 log_t", "_s_densenet169 log_t^2", "_s_densenet201 log_t", "_s_densenet201 log_t^2", "_s_googlenet log_t", "_s_googlenet log_t^2", "_s_resnet152 log_t", "_s_resnet152 log_t^2", "_s_vgg13 log_t", "_s_vgg13 log_t^2", "_s_vgg16 log_t", "_s_vgg16 log_t^2", "_s_vgg19 log_t", "_s_vgg19 log_t^2", "_s_vit-b-16 log_t", "_s_vit-b-16 log_t^2", "_s_vit-b-32 log_t", "_s_vit-b-32 log_t^2", "_s_vit-h-14 log_t", "_s_vit-h-14 log_t^2", "_s_vit-l-16 log_t", "_s_vit-l-16 log_t^2", "_s_vit-l-32 log_t", "_s_vit-l-32 log_t^2"], "model_params": [4.53413724899292, 4.041447162628174, 5.239640235900879, 3.7963831424713135, 4.01073694229126, 3.947139263153076, 3.867375135421753, 2.80039644241333, 4.858762741088867, -9.514336585998535, -3.44964599609375, 3.4373276233673096, -2.9054508209228516, 0.0, 0.0, -3.022820472717285, 10.049932479858398, 0.1929219514131546, 0.0, 0.0, -0.6046064496040344, -0.9322256445884705, 1.3660039901733398, 0.0, 0.0, 0.2948245704174042, -2.0058634281158447, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.21618717908859253, -4.136499404907227, -0.6836631298065186, 0.9377488493919373, 0.2952524423599243, 0.7679898142814636, 0.3058469593524933, 0.25647953152656555, -0.10988038778305054, 0.8402564525604248, 0.2945016026496887, 2.494960308074951, 0.9036676287651062, 2.9217491149902344, 1.042955756187439, 2.6433818340301514, 0.8187727332115173, 1.4187272787094116, 0.7772136926651001, -0.4575100541114807, -0.49121853709220886, 2.9431467056274414, -0.5280545949935913, 1.885624885559082, -0.10109750926494598, 3.5307095050811768, 1.433680772781372], "model_aic": 639429.5756561653}
"""),
                TaskCompletionModel.deserialize("""
{"stratification_columns": ["conf_model_name"], "constant_feature_columns": ["conf_task_count"], "variable_feature_columns": ["res_net_read", "res_net_write", "res_cpu", "res_memory"], "ignored_feature_columns": ["res_net_read"], "standardize": true, "means": [0.4619435667991638, 14.736695289611816, 0.0, 46778088.0, 85.1463394165039, 744487616.0], "stds": [0.6270424127578735, 17.056474685668945, 1.0, 70234840.0, 102.2030258178711, 777391360.0], "polynomial_degree": 2, "polynomial_powers": [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], "model_strata": ["densenet169", "densenet201", "googlenet", "resnet152", "vgg13", "vgg16", "vgg19", "vit-b-16", "vit-b-32", "vit-h-14", "vit-l-16", "vit-l-32"], "model_stratum_dummy_names": ["_s_densenet169", "_s_densenet201", "_s_googlenet", "_s_resnet152", "_s_vgg13", "_s_vgg16", "_s_vgg19", "_s_vit-b-16", "_s_vit-b-32", "_s_vit-h-14", "_s_vit-l-16", "_s_vit-l-32"], "model_param_names": ["_s_densenet169", "_s_densenet201", "_s_googlenet", "_s_resnet152", "_s_vgg13", "_s_vgg16", "_s_vgg19", "_s_vit-b-16", "_s_vit-b-32", "_s_vit-h-14", "_s_vit-l-16", "_s_vit-l-32", "conf_task_count", "res_net_read", "res_net_write", "res_cpu", "res_memory", "log_t conf_task_count", "log_t res_net_read", "log_t res_net_write", "log_t res_cpu", "log_t res_memory", "conf_task_count^2", "conf_task_count res_net_read", "conf_task_count res_net_write", "conf_task_count res_cpu", "conf_task_count res_memory", "res_net_read^2", "res_net_read res_net_write", "res_net_read res_cpu", "res_net_read res_memory", "res_net_write^2", "res_net_write res_cpu", "res_net_write res_memory", "res_cpu^2", "res_cpu res_memory", "res_memory^2", "_s_densenet169 log_t", "_s_densenet169 log_t^2", "_s_densenet201 log_t", "_s_densenet201 log_t^2", "_s_googlenet log_t", "_s_googlenet log_t^2", "_s_resnet152 log_t", "_s_resnet152 log_t^2", "_s_vgg13 log_t", "_s_vgg13 log_t^2", "_s_vgg16 log_t", "_s_vgg16 log_t^2", "_s_vgg19 log_t", "_s_vgg19 log_t^2", "_s_vit-b-16 log_t", "_s_vit-b-16 log_t^2", "_s_vit-b-32 log_t", "_s_vit-b-32 log_t^2", "_s_vit-h-14 log_t", "_s_vit-h-14 log_t^2", "_s_vit-l-16 log_t", "_s_vit-l-16 log_t^2", "_s_vit-l-32 log_t", "_s_vit-l-32 log_t^2"], "model_params": [6.793107032775879, 6.1921000480651855, 8.049272537231445, 5.311357021331787, 3.1423797607421875, 2.7272095680236816, 2.403982639312744, 3.8369979858398438, 7.929607391357422, -16.465219497680664, -5.653622150421143, 1.4980448484420776, -7.032759666442871, 0.0, 2.406296968460083, 2.772109270095825, 13.538697242736816, 0.20457006990909576, 0.0, 2.0295708179473877, -0.3107835054397583, 4.647342205047607, 2.398761749267578, 0.0, -1.4611700773239136, -1.3425542116165161, 0.5671573877334595, 0.0, 0.0, 0.0, 0.0, 0.8238736391067505, -0.15992170572280884, 1.0953010320663452, 0.3697469234466553, -2.3551385402679443, 0.034861866384744644, 6.383235931396484, 0.3299531638622284, 6.41298770904541, 0.33964818716049194, 6.181217193603516, 0.30203306674957275, 6.640302658081055, 0.3403913974761963, 6.778543472290039, 0.45760712027549744, 6.870018005371094, 0.4721982479095459, 6.90427303314209, 0.4801425039768219, 6.955026626586914, 0.3561911880970001, 5.77972412109375, 0.4224402904510498, 8.892152786254883, 1.2191956043243408, 8.290572166442871, 0.6262850165367126, 6.948611736297607, 0.5092968940734863], "model_aic": 186429.6005374225}
"""),
            ],
        ]
    )
    def task_completion_models(self, request):
        return request.param

    @pytest.fixture
    def simulator(self, n_series, task_completion_models):
        return VectorPLRSimulator(task_completion_models=task_completion_models)

    def test_correctness_of_output(self, simulator: VectorPLRSimulator, task_completion_models: list[TaskCompletionModel], n_series: int):
        n_timesteps = 1000

        task_count = torch.randint(
            low=1, high=100, size=(n_timesteps, n_series))
        bandwidth_in = torch.normal(
            mean=100_000_000., std=10_000_000., size=(n_timesteps, n_series))
        bandwidth_out = torch.normal(
            mean=100_000_000., std=10_000_000., size=(n_timesteps, n_series))
        cpu = torch.normal(mean=100., std=30., size=(n_timesteps, n_series))
        memory = torch.normal(
            mean=2 * 1024 * 1024 * 1024., std=500 * 1024 * 1024., size=(n_timesteps, n_series))
        time = torch.randint(
            low=1, high=100, size=(n_timesteps, n_series)).float()
        strata_idx = torch.randint(
            0, len(task_completion_models[0].model_strata), size=(n_timesteps, n_series))

        def run(simulator, model_idx, bandwidth_in, bandwidth_out, cpu, memory):
            resources = torch.cat([
                bandwidth_in.unsqueeze(1),
                bandwidth_out.unsqueeze(1),
                cpu.unsqueeze(1),
                memory.unsqueeze(1),
            ], dim=1)

            # generator = self._get_test_output_generator(
            #     models=task_completion_models,
            #     model_idx=model_idx,
            #     const_covariates=[task_count],
            #     var_covariates=[bandwidth_in, bandwidth_out, cpu, memory],
            #     strata_idx=strata_idx,
            #     aggregation_window=window,
            #     n_timesteps=n_timesteps,
            #     n_series=n_series,
            # )

            for t in range(n_timesteps):
                simulated_survival_probabilities = simulator.simulate(
                    const_covariates=task_count[t][None],
                    var_covariates=resources[t],
                    time=time[t],
                    model_idx=model_idx[t],
                    strata_idx=strata_idx[t],
                    mask=torch.ones(n_series, dtype=torch.bool),
                )

                # test_survival_probabilities = next(generator)
                test_survival_probabilities = self._calc_test_survival_probabilities(
                    models=task_completion_models,
                    model_idx=model_idx[t],
                    const_covariates=task_count[t][None],
                    var_covariates=resources[t],
                    time=time[t],
                    strata_idx=strata_idx[t],
                    n_series=n_series,
                )

                assert torch.allclose(simulated_survival_probabilities,
                                      test_survival_probabilities, atol=1e-5)
                assert torch.all((simulated_survival_probabilities >= 0.0) & (
                    simulated_survival_probabilities <= 1.0))

        run(
            simulator=deepcopy(simulator),
            model_idx=torch.full((n_timesteps, n_series), fill_value=0),
            bandwidth_in=bandwidth_in,
            bandwidth_out=torch.zeros_like(bandwidth_out),
            cpu=cpu,
            memory=memory,
        )
        run(
            simulator=deepcopy(simulator),
            model_idx=torch.full((n_timesteps, n_series), fill_value=1),
            bandwidth_in=torch.zeros_like(bandwidth_in),
            bandwidth_out=torch.zeros_like(bandwidth_out),
            cpu=cpu,
            memory=memory,
        )

        model_idx = torch.where(
            torch.arange(n_timesteps * n_series) % 2 == 0,
            torch.full((n_timesteps * n_series,), fill_value=0),
            torch.full((n_timesteps * n_series,), fill_value=1),
        ).view(n_timesteps, n_series)

        run(
            simulator=deepcopy(simulator),
            model_idx=model_idx,
            bandwidth_in=torch.where(
                model_idx == 0, bandwidth_in, torch.zeros_like(bandwidth_in)),
            bandwidth_out=torch.zeros_like(bandwidth_out),
            cpu=cpu,
            memory=memory,
        )

    def test_mask(self, simulator: VectorPLRSimulator, n_series: int):
        """Masked series (mask=False) should always return survival_probability=1."""
        task_count = torch.randint(low=1, high=100, size=(n_series,))
        resources = torch.normal(mean=100., std=10., size=(4, n_series))
        time = torch.randint(low=1, high=100, size=(n_series,)).float()
        strata_idx = torch.zeros(n_series, dtype=torch.long)
        model_idx = torch.zeros(n_series, dtype=torch.long)

        mask = torch.zeros(n_series, dtype=torch.bool)
        survival_probabilities = simulator.simulate(
            const_covariates=task_count[None],
            var_covariates=resources,
            time=time,
            model_idx=model_idx,
            strata_idx=strata_idx,
            mask=mask,
        )
        assert torch.all(survival_probabilities == 1.0)

    def test_time_zero_or_negative(self, simulator: VectorPLRSimulator, n_series: int):
        """Series with time <= 0 should always return survival_probability=1."""
        task_count = torch.randint(low=1, high=100, size=(n_series,))
        resources = torch.normal(mean=100., std=10., size=(4, n_series))
        time = torch.zeros(n_series)
        strata_idx = torch.zeros(n_series, dtype=torch.long)
        model_idx = torch.zeros(n_series, dtype=torch.long)
        mask = torch.ones(n_series, dtype=torch.bool)

        survival_probabilities = simulator.simulate(
            const_covariates=task_count[None],
            var_covariates=resources,
            time=time,
            model_idx=model_idx,
            strata_idx=strata_idx,
            mask=mask,
        )
        assert torch.all(survival_probabilities == 1.0)

    def test_output_in_valid_range(self, simulator: VectorPLRSimulator, n_series: int):
        """Survival probabilities must always lie in [0, 1]."""
        n_timesteps = 100
        task_count = torch.randint(
            low=1, high=100, size=(n_timesteps, n_series))
        resources = torch.normal(
            mean=100., std=10., size=(n_timesteps, 4, n_series))
        time = torch.randint(low=1, high=500, size=(
            n_timesteps, n_series)).float()
        strata_idx = torch.randint(
            0, len(simulator._example_model.model_strata), size=(n_timesteps, n_series))
        model_idx = torch.zeros(n_timesteps, n_series, dtype=torch.long)
        mask = torch.ones(n_series, dtype=torch.bool)

        for t in range(n_timesteps):
            survival_probabilities = simulator.simulate(
                const_covariates=task_count[t][None],
                var_covariates=resources[t],
                time=time[t],
                model_idx=model_idx[t],
                strata_idx=strata_idx[t],
                mask=mask,
            )
            assert torch.all((survival_probabilities >= 0.0) &
                             (survival_probabilities <= 1.0))

    # Test helper functions

    def test_standardization(self, n_series, task_completion_models: list[TaskCompletionModel]):
        model = task_completion_models[0]
        n_timesteps = 100
        n_features = model.means.shape[0]

        var = torch.normal(mean=100_000_000., std=10_000_000.,
                           size=(n_timesteps, n_features, n_series))

        for t in range(n_timesteps):
            features = var[t]
            standardized_features = self._standardize_features(model, features)

            test_features = torch.zeros_like(standardized_features)

            for i in range(n_features):
                for j in range(n_series):
                    test_features[i, j] = (
                        features[i, j] - model.means[i]) / model.stds[i]

            assert torch.all(standardized_features == test_features)

    def test_poly_feature_calc(self, task_completion_models: list[TaskCompletionModel], n_series):
        model = task_completion_models[0]
        n_timesteps = 100
        n_features = model.polynomial_powers.shape[1]

        features_over_t = torch.normal(
            mean=100., std=50., size=(n_timesteps, n_features, n_series))

        for t in range(n_timesteps):
            features = features_over_t[t]

            poly_features = self._compute_poly_feature(model, features)

            test_features = torch.zeros_like(poly_features)

            for out_i in range(model.polynomial_powers.shape[0]):
                for in_i in range(model.polynomial_powers.shape[1]):
                    power = model.polynomial_powers[out_i, in_i]

                    if torch.all(test_features[out_i] == 0):
                        test_features[out_i] = features[in_i] ** power
                    else:
                        test_features[out_i] *= features[in_i] ** power

            assert torch.all(poly_features == test_features)

    # Helpers

    def _calc_test_survival_probabilities(
        self,
        models: list[TaskCompletionModel],
        model_idx: torch.Tensor,
        const_covariates: torch.Tensor,
        var_covariates: torch.Tensor,
        time: torch.Tensor,
        strata_idx: torch.Tensor,
        n_series: int,
    ) -> torch.Tensor:
        """
        Reference implementation of VectorPLRSimulator.simulate, computed
        series-by-series for clarity.

        const_covariates: [n_const, n_series]
        var_covariates:   [n_var, n_series]
        time:             [n_series]
        model_idx:        [n_series]
        strata_idx:       [n_series]
        """
        n_strata = len(models[0].model_strata)
        test_survival_probabilities = []

        for i in range(n_series):
            model = models[model_idx[i].item()]
            t = time[i].item()

            # time <= 0 → survival probability is 1 (not yet started)
            if t <= 0:
                test_survival_probabilities.append(1.0)
                continue

            # Build base feature vector [log_t, const_covariates, var_covariates]
            # shape: [n_base]
            log_t = torch.log(torch.tensor(t).clamp(min=1))
            base_features = torch.cat([
                log_t.unsqueeze(0),
                const_covariates[:, i],
                var_covariates[:, i],
            ])

            # Standardize
            if model.standardize:
                base_features = (base_features - model.means) / model.stds

            # Append one-hot stratum dummy
            stratum = strata_idx[i].item()
            dummy = torch.zeros(n_strata)
            dummy[stratum] = 1.0
            feature_vector = torch.cat([base_features, dummy])

            # Polynomial transform
            feature_vector = self._compute_poly_feature(
                model, feature_vector.unsqueeze(-1)).squeeze(-1)

            # Linear predictor → sigmoid → survival probability
            linear_predictor = torch.dot(model.model_params, feature_vector)
            event_probability = torch.sigmoid(linear_predictor)
            survival_probability = 1.0 - event_probability

            test_survival_probabilities.append(survival_probability.item())

        return torch.tensor(test_survival_probabilities)

    def _standardize_features(self, model: TaskCompletionModel, features: torch.Tensor):
        return (features - model.means.unsqueeze(-1)) / model.stds.unsqueeze(-1)

    def _compute_poly_feature(self, model: TaskCompletionModel, features: torch.Tensor):
        return torch.prod(
            features.T.unsqueeze(1) ** model.polynomial_powers.unsqueeze(0), dim=-1,
        ).T
