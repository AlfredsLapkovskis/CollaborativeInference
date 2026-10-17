for i in {0..19}; do
    echo "Fitting x storage consumption model"
    python -m scripts.resource_consumption.fit_resource_consumption_model \
        --experiment fit_cons_x_download_res_storage_$i
done
