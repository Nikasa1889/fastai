#!/bin/bash
docker run --runtime=nvidia -d -v ${PWD}:/fastai -v /datasets:/fastai/data -p 8880:8888 fastai:latest
