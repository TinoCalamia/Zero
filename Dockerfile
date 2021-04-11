FROM public.ecr.aws/lambda/python:3.8 as base
FROM base AS deploy

LABEL maintainer="calamia.tino@gmail.com" service=zero

# Install python packages
COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade --upgrade-strategy=eager -r requirements.txt
RUN yum install -y wget yum tar gzip gcc-c++ libcurl-devel make

RUN export CC=gcc64
RUN export CXX=g++64

# CMake
#RUN wget -O cmake.sh https://github.com/Kitware/CMake/releases/download/v3.18.1/cmake-3.18.1-Linux-x86_64.sh && \
#   sh ./cmake.sh --prefix=/usr/local --skip-license

# Copy .src folder file to workdir /app
COPY . ./
CMD ["app.lambda_handler"]


