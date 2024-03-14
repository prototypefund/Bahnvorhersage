# DNS and pods internet access

I have had some trouble using microk8s dns to give pods access to the internet. The following helped to get the DNS running again after it randomly broke:

```sh
microk8s enable dns:192.168.178.1
```

## Further info
- k8s dns troubleshooting: https://kubernetes.io/docs/tasks/administer-cluster/dns-debugging-resolution/#known-issues
- my issue on GitHub: https://github.com/canonical/microk8s/issues/4459