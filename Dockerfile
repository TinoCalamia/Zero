FROM python:latest as python

LABEL maintainer="calamia.tino@gmail.com" service=da_vinci

WORKDIR /app

# Install python packages
COPY requirements.txt /app

RUN pip install --no-cache-dir --upgrade --upgrade-strategy=eager -r requirements.txt

RUN apt-get update && \
    apt-get install -yqq --no-install-recommends git && \
    pip install --upgrade -q black && \
    pip install -U jupyterlab==1.2.0 && \
    pip install seaborn nb_black pyarrow

RUN curl -sL https://deb.nodesource.com/setup_10.x | bash -&& \
    apt-get install -yqq --no-install-recommends nodejs && \
    jupyter labextension install jupyterlab-spreadsheet

# Copy .src folder file to workdir /app
COPY . /app