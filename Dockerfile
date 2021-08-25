FROM public.ecr.aws/lambda/python:3.8 as base
FROM base AS deploy

LABEL maintainer="calamia.tino@gmail.com" service=zero

# Install python packages
COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade --upgrade-strategy=eager -r requirements.txt

# Copy .src folder file to workdir /app
COPY . ./
CMD ["app.lambda_handler"]


