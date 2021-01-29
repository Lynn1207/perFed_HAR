taskset --cpu-list 0 python ./cnnHAR_train.py 1 &
taskset --cpu-list 1 python ./cnnHAR_train.py 2 

