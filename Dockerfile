FROM nvidia/cuda:10.1-devel
RUN groupadd -g 1001 user
RUN useradd -u 1001 -g 1001 -ms /bin/bash  user
RUN mkdir /deepdiva
RUN chown -R user:user /deepdiva
RUN apt-get update && apt-get install -y wget
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash ./miniconda.sh -b -p /opt/conda && \
    rm ./miniconda.sh
ENV PATH /opt/conda/bin:$PATH
ENV PYTHONPATH /deepdiva:$PYTHONPATH
USER user
WORKDIR /deepdiva
ADD environment.yml environment.yml
RUN conda env create -f environment.yml
ENV PATH /home/user/.conda/envs/deepdiva/bin/:$PATH
ADD . /deepdiva
USER root
RUN chmod +x classify_image.sh
USER user
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["python", "template/RunMe.py"]