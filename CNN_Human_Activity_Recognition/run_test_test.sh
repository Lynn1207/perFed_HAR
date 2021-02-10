taskset --cpu-list 0 python ./cnnHAR_train.py 1 &
taskset --cpu-list 1 python ./cnnHAR_train.py 2 &
taskset --cpu-list 2 python ./cnnHAR_train.py 3 &
taskset --cpu-list 3 python ./cnnHAR_train.py 4 &
taskset --cpu-list 4 python ./cnnHAR_train.py 5 &
taskset --cpu-list 5 python ./cnnHAR_train.py 6 
