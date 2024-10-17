## Introduction

### What is Kubernetes?
Kubernetes (K8s) is an open-source container orchestration platform that automates the deployment, scaling, and management of containerized applications. It is designed to facilitate the management of large-scale applications with high availability.

### What is Helm?
Helm is a package manager for Kubernetes that helps you define, install, and upgrade even the most complex Kubernetes applications. Helm uses a packaging format called charts, which are collections of files that describe a related set of Kubernetes resources.


## Deployment with Helm
To deploy your application, use the following command:

`helm upgrade --install <release-name> <chart-path>`

For the istance `my-memgpt`:
```bash
helm upgrade --install my-memgpt .
```

Then, get the application URL by running these commands:
```bash
export POD_NAME=$(kubectl get pods --namespace default -l "app.kubernetes.io/name=memgpt,app.kubernetes.io/instance=my-memgpt" -o jsonpath="{.items[0].metadata.name}")
export CONTAINER_PORT=$(kubectl get pod --namespace default $POD_NAME -o jsonpath="{.spec.containers[0].ports[0].containerPort}")
echo "Visit http://127.0.0.1:8080 to use your application"
kubectl --namespace default port-forward $POD_NAME 8080:$CONTAINER_PORT
```

Go to the browser, log in with `password`, and enjoy!
