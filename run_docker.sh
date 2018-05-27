#!/bin/bash
docker run --runtime=nvidia -v ${PWD}:/fastai -v /datasets:/fastai/data -p 8880:8888 nikasa/fastai:jupyter_ex
