FROM continuumio/miniconda3:4.7.12

# ENV HOME /app

# # ENV GRB_LICENSE_FILE=/gurobi.lic
# COPY environment.yml /environment.yml

# RUN conda env create mip -f /environment.yml && conda clean -a
# RUN conda  env update conda -f environment.yml

# ENV PATH=/opt/conda/envs/r3/bin:$PATH

# WORKDIR $HOME

# # Set up working directory
# #Add repository content to working directory
# COPY . $HOME

# RUN echo "source activate ml" > ~/.bashrc