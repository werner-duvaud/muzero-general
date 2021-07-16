FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime as base

RUN apt update && \
    apt install -y make git

FROM base as second

# OpenAI Gym cartpool environment
RUN apt update -y && \
    apt install -y xvfb ffmpeg freeglut3-dev python-opengl x11-utils

# Need for video recording on cartpole enviroment
RUN conda install -y x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN pip install jupyter twine bump2version

FROM second

WORKDIR /work

# RUN jupyter contrib nbextension install --user && \
#     jupyter nbextension enable autoscroll/main

CMD [ "make", "notebook" ] 