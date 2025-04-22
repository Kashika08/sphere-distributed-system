docker rmi prophet-model
cd ../data-handling/ml-engine/prophet
pwd
docker build -t prophet-model .
docker run --rm -it   -v "$(pwd)/../ml_data.log:/app/ml_data.log"   -v "$(pwd)/../ml_predictions.log:/app/ml_predictions.log" prophet-model

