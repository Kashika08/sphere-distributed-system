# SPHERE: Scalable Proactive Handling for Efficient Resource Expansion
This repository contains a distributed system implementation that integrates Kafka messaging, Redis caching, Prometheus monitoring, and a Sock Shop microservices application. The system includes load testing capabilities with Locust and autoscaling through our machine learning models.

## System Overview

The Sphere Distributed System is designed to demonstrate microservices architecture, distributed data processing, and machine learning capabilities within a Kubernetes environment. It uses the Sock Shop microservices application as a benchmark to evaluate system performance and scalability.

## System Setup Instructions
## Prerequisites

-   Docker
-   Kubernetes CLI (kubectl)
-   Minikube
-   Helm 3
-   Git

## Technologies
- C++
- Kubernetes
- Kafka
- Prometheus
- Redis
- Machine Learning
- Minikube
- Docker
- Helm
- Git
- CMake
- Python
- Shell

> [!TIP]
> Certain commands below will block the SSH terminal. Having multiple terminal login helps! 

## Step 1: Clone the Repository

	git clone https://github.com/Kashika08/sphere-distributed-system.git
	cd sphere-distributed-system

## Step 2: Start Minikube

	minikube start --cpus=4 --memory=8192 --driver=docker 

## Step 3: Deploy Sock Shop Microservices

	git clone https://github.com/microservices-demo/microservices-demo.git
	kubectl apply -f microservices-demo/deploy/kubernetes/complete-demo.yaml -n sock-shop

This will deploy the Sock Shop microservices
[Sock Shop Microservice Benchmark](https://github.com/microservices-demo/microservices-demo/tree/master)

## Step 4: Install Prometheus Monitoring Stack

	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
	helm repo update
	helm install prometheus prometheus-community/kube-prometheus-stack -f deployments/prometheus/values.yaml

## Step 5: Deploy Redis Cluster
    docker run -d --name redis-server -p 6379:6379 redis

## Step 6: Setup Kafka
	docker pull apache/kafka:3.9.0
	docker run -p 9092:9092 apache/kafka:3.9.0

## Step 7: Deploy the ML Model Service

	cd data-handling/ml-engine/holt-winters
	docker build -t holt-winters .

## Step 8: Data Generation

### Run Locust
```bash
cd data-generation
locust -f locustfile.py --host=http://192.168.49.2:30001
```

### Monitor System Performance

	kubectl port-forward svc/prometheus-operated 9090:9090 -n sock-shop
    Access Prometheus at [http://localhost:9090](http://localhost:9090/) to monitor system performance.

## Step 9: Data Collection

### Fetch Pod Metrics (Producer, topic :: metrics)
```bash
cd data-collection
mkdir build
bash build.sh
```

## Step 10: Data Handling

### Metrics Consumer (Consumer, topic :: metrics)
```bash
cd data-handling/ml-engine/metrics-consumer
mkdir build
bash build.sh
```

### ML model (Consumer, topic :: metrics)
Run the ML model built
```bash
cd data-handling/ml-engine/holt-winters
docker run --rm -it   -v "$(pwd)/../ml_data.log:/app/ml_data.log"   -v "$(pwd)/../ml_predictions.log:/app/ml_predictions.log" holt-winters
```

### Predictions Producer (Producer, topic :: predictions)
```bash
cd data-handling/ml-engine/metrics-consumer
mkdir build
bash build.sh
```

## Step 11: Redis
### Redis Consumer (Consumer, topic :: predictions)
```bash
cd ~/sphere-distributed-system
docker build -t redis-consumer -f redis-consumer/Dockerfile .
docker run --network="host" redis-consumer

docker exec -it redis-server redis-cli
CONFIG SET notify-keyspace-events KEA
```

## Autoscaler
```bash
cd autoscaler
python3 autoscaler.py
```

## Pod Status Checking

	kubectl get pods -n sock-shop

## Logs Retrieval

	kubectl logs <pod-name> -n <namespace>

## Resource Utilization

	kubectl top pods -n sock-shop