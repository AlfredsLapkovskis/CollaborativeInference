for i in {0..19}; do
    echo "Fitting y execution memory consumption model"
    python -m scripts.resource_consumption.fit_resource_consumption_model \
        --experiment fit_cons_y_execution_res_memory_$i
done
