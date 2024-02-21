# db-rest uses an alpine image whitch does currently does not work with kubernetes
# https://stackoverflow.com/questions/65181012/does-alpine-have-known-dns-issue-within-kubernetes
# Therefore we use copy the db-rest image into a debian based image

DOCKER_BUILDKIT=1 docker build -f k8s/Dockerfile.db_rest  k8s -t trainconnectionprediction/db-rest:6

docker push trainconnectionprediction/db-rest:6