for i in {0..19}; do
    echo "Fitting x net_read consumption model"
    python -m scripts.resource_consumption.fit_resource_consumption_model \
        --experiment fit_cons_x_download_res_net_read_$i
done
