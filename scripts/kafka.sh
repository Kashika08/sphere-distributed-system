#!/bin/bash

container_id=$(docker ps -q --filter ancestor=apache/kafka:3.9.0)
echo "Kafka at $container_id"

if [ -n "$container_id" ]; then
  docker stop $container_id
  docker rm $container_id
fi

docker run -dp 9092:9092 apache/kafka:3.9.0
