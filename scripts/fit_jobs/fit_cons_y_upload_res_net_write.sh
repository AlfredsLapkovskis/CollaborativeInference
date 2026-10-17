for i in {0..19}; do
    echo "Fitting y upload net_write consumption model"
    python -m scripts.resource_consumption.fit_resource_consumption_model \
        --experiment fit_cons_y_upload_res_net_write_$i
done
