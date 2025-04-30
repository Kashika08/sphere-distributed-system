#!/bin/bash
cd ~/sphere-distributed-system
docker rmi redis-consumer
docker build -t redis-consumer -f redis-consumer/Dockerfile .
docker run -d --network="host" redis-consumer

docker exec -it redis-server redis-cli
CONFIG SET notify-keyspace-events KEA
