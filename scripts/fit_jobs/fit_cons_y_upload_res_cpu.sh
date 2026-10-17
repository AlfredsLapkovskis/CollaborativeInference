for i in {0..19}; do
    echo "Fitting y upload cpu consumption model"
    python -m scripts.resource_consumption.fit_resource_consumption_model \
        --experiment fit_cons_y_upload_res_cpu_$i
done
