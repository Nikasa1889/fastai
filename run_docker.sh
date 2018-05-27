#!/bin/bash
docker run --runtime=nvidia -d -v ${PWD}:/fastai -v /datasets:/fastai/data -v /datasets/fastai/weights:/fastai/weights -p 8880:8888 nikasa/fastai:latest
