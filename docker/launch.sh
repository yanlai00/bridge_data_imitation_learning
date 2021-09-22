# instructions for running this the first time:
# pull the docker image from docker hub: run ` docker pull <>:latest `
docker run --gpus all \
                   -v /:/parent \
                   --name=${DOCKER_NAME} \
-e EXP=/parent${EXP} \
-e DATA=/parent${DATA} \
-t -d \
--shm-size 16G \
febert/spt_image:latest \
/bin/bash
docker exec ${DOCKER_NAME} /bin/bash -c " "
