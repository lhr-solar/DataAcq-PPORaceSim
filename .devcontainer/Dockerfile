FROM mcr.microsoft.com/devcontainers/anaconda

# System packages 
RUN apt-get update && apt-get install -yq curl wget jq vim ffmpeg libsm6 libxext6

RUN conda update -y conda
RUN conda init

RUN conda install -c anaconda -y python=3.9.16
RUN conda install -c anaconda -y tensorboard
RUN conda install -c conda-forge -y mkl=2023.2.0
RUN conda install -c conda-forge -y irrlicht=1.8.5
RUN conda install -c nvidia/label/cuda-11.7.0 -y cuda-toolkit
RUN conda install -c conda-forge -y glfw

RUN pip install stable-baselines3[extra]==2.1.0
RUN pip install gymnasium==0.29.1
RUN pip install pygame==2.5.1
RUN pip install scipy==1.10.1
RUN pip install liionpack
RUN pip install splines

# RUN curl -LO 
RUN conda install -y "https://anaconda.org/projectchrono/pychrono/9.0.0/download/linux-64/pychrono-9.0.0-py39_4853.tar.bz2"

RUN pip install opencv-python==4.7.0.72
RUN pip uninstall -y pyqt5
RUN pip install lmfit pvlib

CMD [ "/bin/bash", ]