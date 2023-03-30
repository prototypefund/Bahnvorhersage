# This Dockerfile should replace many other dockerfiles
# in this project by building an base that can run all
# python scripts in this repo

# DOCKER_BUILDKIT=1 docker build . -t bahnvorhersage
# docker run -v $(pwd)/config.py:/usr/src/app/config.py -v $(pwd)/cache:/usr/src/app/cache bahnvorhersage python3 python/station_finder.py

FROM osgeo/proj:9.2.0 as proj
# This is needed as cartopy dependency

FROM python:3.11 AS compile-image

# Install dependencies (libgeos in order for cartopy to work)
RUN apt-get update
RUN apt-get install -y gcc build-essential 
RUN apt-get install -y libgeos-dev

COPY --from=proj  /usr/share/proj/ /usr/share/proj/
COPY --from=proj  /usr/include/ /usr/include/
COPY --from=proj  /usr/bin/ /usr/bin/
COPY --from=proj  /usr/lib/ /usr/lib/

## Virtualenv
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

## Add and install requirements
RUN pip install --upgrade pip 
# Install shapely without binary in order to link to the correct geos lib
# RUN pip install shapely cartopy --no-binary shapely --no-binary cartopy
RUN pip install shapely --no-binary shapely
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

## runtime-image
FROM python:3.11 AS runtime-image

COPY --from=proj  /usr/share/proj/ /usr/share/proj/
COPY --from=proj  /usr/include/ /usr/include/
COPY --from=proj  /usr/bin/ /usr/bin/
COPY --from=proj  /usr/lib/ /usr/lib/

# Install dependencies (libgeos in order for cartopy to work)
RUN apt-get update
RUN apt-get install -y libgeos-dev

## Virtualenv
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

## Copy Python dependencies from build image
COPY --from=compile-image /opt/venv /opt/venv
# It seems to be overkill to reinstall shapely here, but this
# fixed the docker build just randomly crashing withour any error
RUN pip install shapely --no-binary shapely --force

## set working directory
WORKDIR /usr/src/app

## Add User (a security measure)
# We have to set a static user id, so that the user can read the files in virtual volumes
# We use system accounts, but it's just symbolic
# RUN addgroup --system --gid 420 tcp && adduser --system --no-create-home --uid 420 --gid 420 tcp

## Add webserver and librays
COPY ./data_analysis/ /usr/src/app/data_analysis
COPY ./database/ /usr/src/app/database/
COPY ./helpers/ /usr/src/app/helpers/
COPY ./ml_models/ /usr/src/app/ml_models
COPY ./model_analysis/ /usr/src/app/ml_models
COPY ./python/ /usr/src/app/python
COPY ./rtd_crawler/ /usr/src/app/rtd_crawler
COPY ./tests/ /usr/src/app/tests
COPY ./update_butler/ /usr/src/app/update_butler
COPY ./webserver/ /usr/src/app/webserver/
COPY __init__.py /usr/src/app/__init__.py
COPY ./webserverconfig.py /usr/src/app/webserverconfig.py

## Switch to non-root user
# for some reason doing this before the copy results in weird permissions # && chmod -R 775 /usr/src/app/
# RUN chown -R tcp:tcp /usr/src/app/
RUN addgroup --system --gid 420 tcp && adduser --system --no-create-home --uid 420 --gid 420 tcp
USER tcp

## Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PATH="/opt/venv/bin:$PATH"
ENV MPLCONFIGDIR="/tmp/matplotlib"


