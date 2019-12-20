FROM continuumio/miniconda3:4.7.12

RUN apt-get update && apt-get -y install build-essential

ENV HOME /app
WORKDIR $HOME
COPY . $HOME  

RUN conda env create ml -f environment.yml && conda clean -a
ENV PATH=/opt/conda/envs/ml/bin:$PATH

RUN echo "source activate ml" > ~/.bashrc