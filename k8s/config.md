Die Config (config.py) läuft in dem Cluster über eine ConfigMap, die den Inhalt der config.py erhält, und wie folgt ausschaut:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: config
  namespace: default
data:
  config.py: |
    Hier Inhalt von config.py (schön sauber eingerückt)
```