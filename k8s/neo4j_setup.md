Hier ein paar Notizen dazu, wie ich neo4j aufgesetzt habe:

Guide: https://neo4j.com/docs/operations-manual/current/kubernetes/quickstart-standalone/

Setup Helm
```sh
sudo snap install helm --classic
helm repo add neo4j https://helm.neo4j.com/neo4j
helm repo update
```

Setup Volume
```sh
mkdir /mnt/neo4j-storage
sudo useradd -u 7474 neo4j
sudo chown -R neo4j:neo4j neo4j-storage
```

Install neo4j
```sh
kubectl create namespace neo4j
kubectl config set-context --current --namespace=neo4j
helm install my-neo4j-release neo4j/neo4j --namespace neo4j -f neo4j.values.yaml
```

Enable Metallb for neo4j load balancer to work:
```sh
microk8s enable metallb 
# Enter IP range: 192.168.200.1-192.168.200.255
```