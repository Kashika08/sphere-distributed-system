rm ../ml_data.log
touch ../ml_data.log
cd build
cmake ..
make
./ml_handler
