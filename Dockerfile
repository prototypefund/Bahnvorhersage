# This Dockerfile should replace many other dockerfiles
# in this project by building an base that can run all
# python scripts in this repo

# DOCKER_BUILDKIT=1 docker build . -t bahnvorhersage
# docker run -v $(pwd)/config.py:/usr/src/app/config.py -v $(pwd)/cache:/usr/src/app/cache bahnvorhersage python3 -m crawler.plan
FROM python:3.11-slim

## Virtualenv
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

## Add and install requirements
RUN pip install --upgrade pip 
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

## set working directory
WORKDIR /usr/src/app

## Add webserver and librays
COPY ./api/ /usr/src/app/api/
COPY ./crawler/ /usr/src/app/crawler/
COPY ./data_analysis/ /usr/src/app/data_analysis/
COPY ./database/ /usr/src/app/database/
COPY ./gtfs/ /usr/src/app/gtfs/
COPY ./helpers/ /usr/src/app/helpers/
COPY ./ml_models/ /usr/src/app/ml_models/
COPY ./parser/ /usr/src/app/parser/
COPY ./python/ /usr/src/app/python/
COPY ./router/ /usr/src/app/router/
COPY ./webserver/ /usr/src/app/webserver/
COPY __init__.py /usr/src/app/__init__.py
COPY ./webserverconfig.py /usr/src/app/webserverconfig.py

## Switch to non-root user
RUN addgroup --system --gid 420 tcp && adduser --system --no-create-home --uid 420 --gid 420 tcp
USER tcp

## Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PATH="/opt/venv/bin:$PATH"
ENV MPLCONFIGDIR="/tmp/matplotlib"

EXPOSE 5000

