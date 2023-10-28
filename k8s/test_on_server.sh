# Run this script to test your changes on next.trainconnectionprediction.de

DOCKER_BUILDKIT=1 docker build . -t trainconnectionprediction/bahnvorhersage:stagging

docker push trainconnectionprediction/bahnvorhersage:stagging

kubectl rollout restart deployment/next-webserver