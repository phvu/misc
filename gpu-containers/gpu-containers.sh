#!/bin/bash

if [ "$1" = "go" ]
then
    nvidia-docker exec -it "$2" bash
else
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    python "${DIR}/gpu-containers.py" $@
fi