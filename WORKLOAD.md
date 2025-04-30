
# SPHERE Distributed System

This repository contains a distributed system implementation that integrates Kafka messaging, Redis caching, Prometheus monitoring, and a Sock Shop microservices application. The system includes load testing capabilities with Locust and autoscaling through our machine learning models.

## System Overview

The Sphere Distributed System is designed to demonstrate microservices architecture, distributed data processing, and machine learning capabilities within a Kubernetes environment. It uses the Sock Shop microservices application as a benchmark to evaluate system performance and scalability.

## Team Responsibilities

## Shubh (snisar)

-   **Kafka Integration**: Setup and configuration of Kafka messaging system
    
-   **Pod Metrics Collection**: Implementation of scripts to fetch pod metrics
    
-   **Load Testing**: Locust script development for performance testing
    
-   **Autoscaling**: HPA (Horizontal Pod Autoscaler) configuration
    
-   **Helper Scripts**: Development of bash scripts to automate tasks
    
-   **Version Control**: Git operations and pull request management
    

## Rajat (rchanda3)

-   **Load Testing**: Locust test execution and reporting
    
-   **Redis**: Setup and configuration of Redis clusters
    
-   **Autoscaling**: Development of custom autoscaler scripts
    
-   **Monitoring**: Prometheus monitoring stack deployment and configuration
    
-   **Version Control**: Git operations and pull request management
    

## Kashika (kmalick)

-   **Environment Setup**: Minikube environment configuration
    
-   **Benchmark Application**: Sock Shop microservices deployment and configuration
    
-   **Machine Learning**: Online retraining ML models development and integration
    
-   **Metrics Collection**: Pod metrics collection and analysis
    
-   **Version Control**: Git operations and pull request management
    

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


## Step 6: Setup Kafka
	docker pull apache/kafka:3.9.0
	docker run -p 9092:9092 apache/kafka:3.9.0

## Step 7: Deploy the ML Model Service

	cd data-handling/ml-engine/holt-winters
	docker build -t holt-winters .
	docker run --rm -it -v "$(pwd)/../ml_data.log:/app/ml_data.log" -v "$(pwd)/../ml_predictions.log:/app/ml_predictions.log" holt-winters

## Running the System

## Start Metrics Collection

	cd data-collection
	mkdir build
	bash build.sh

This script will begin collecting metrics from the front-end pod in the system.

## Run Load Tests with Locust

	cd data-generation
	locust -f locustfile.py --host=http://192.168.49.2:30001

Access the Locust UI through the provided URL to configure and execute load tests.

## Execute Autoscaling Test

	cd autoscaler
	python3 autoscaler.py

This script will initiate autoscaling capabilities on the basis of the predictions from the system.

## Monitor System Performance

	kubectl port-forward svc/prometheus-operated 9090:9090 -n sock-shop
Access Prometheus at [http://localhost:9090](http://localhost:9090/) to monitor system performance.


## Troubleshooting

## Pod Status Checking

	kubectl get pods -n sock-shop

## Logs Retrieval

	kubectl logs <pod-name> -n <namespace>

## Resource Utilization

	kubectl top pods -n sock-shop
