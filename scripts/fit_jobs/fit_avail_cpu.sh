for i in {0..19}; do
    echo "Fitting available resource model $i"
    python -m scripts.available_resources.fit_available_resource_model \
        --experiment fit_avail_cpu_$i
done
