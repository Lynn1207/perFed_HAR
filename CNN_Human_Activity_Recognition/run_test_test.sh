taskset --cpu-list 0 python ./cnnHAR_train.py 1 &
taskset --cpu-list 1 python ./cnnHAR_train.py 2 &
taskset --cpu-list 2 python ./cnnHAR_train.py 3 &
taskset --cpu-list 3 python ./cnnHAR_train.py 4 &
taskset --cpu-list 4 python ./cnnHAR_train.py 5 &
taskset --cpu-list 5 python ./cnnHAR_train.py 6 &
taskset --cpu-list 6 python ./cnnHAR_train.py 7 &
taskset --cpu-list 7 python ./cnnHAR_train.py 8 &
taskset --cpu-list 8 python ./cnnHAR_train.py 9 &
taskset --cpu-list 9 python ./cnnHAR_train.py 10 &
taskset --cpu-list 10 python ./cnnHAR_train.py 11 &
taskset --cpu-list 11 python ./cnnHAR_train.py 12
