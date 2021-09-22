export MJKEY="$(cat ~/.mujoco/mjkey.txt)" \
    && docker-compose \
        -f ./mj_gpu_docker/docker-compose.dev.gpu.yml \
        up \
        -d \
        --no-recreate