import socketserver
import pickle, struct
import sys
from threading import Lock, Thread
import threading
import numpy as np
from sklearn.decomposition import PCA
from keras.utils import to_categorical
import tensorflow.compat.v1 as tf


tf.disable_v2_behavior()
# np.set_printoptions(threshold=np.inf)

NUM_OF_TOTAL_USERS = 6
NUM_OF_WAIT = NUM_OF_TOTAL_USERS
W_DIM = 74848 #l1: 2112; l2: 8288; l3: 74848, l4: 599648; l5: 615038; l6: 615224
inner_iteration = 5
T_thresh = 10

iteration_count = 0
regular = 1e5
alpha = 1*(1e-3)

W = np.zeros((NUM_OF_TOTAL_USERS,W_DIM))
W_avg = np.zeros(W_DIM)
Loss = np.zeros(NUM_OF_TOTAL_USERS)
Loss[Loss<np.inf] = 1e5
Loss_cache = np.zeros(NUM_OF_TOTAL_USERS)
conver_indicator = 1e5
loss_record = np.zeros(1100)
normalized_dloss = np.zeros((NUM_OF_TOTAL_USERS,T_thresh))
update_flag = np.ones(NUM_OF_TOTAL_USERS)

def server_update():
    
    global W_avg, W_avg1_1,W_avg2_1,W_avg3_1, W_avg3_2, W_avg4_1, W_avg4_2, W_avg5_1, W_avg5_2, W_avg5_3, W_avg6_1, W_avg6_2, W_avg6_3
    # print(np.max(W))
    
    
    #W_avg=np.mean(W, axis=0)
    
    W_avg1_1= np.mean(W[0:6,0:2112], axis = 0)
    
    W_avg2_1=np.mean(W[0:6, 2112:8288], axis = 0)
    
    #W_avg3_1=(np.array(W[0][8288:74848])+np.array(W[4][8288:74848])+np.array(W[5][8288:74848]))/3.0
    W_avg3_1=np.mean(W[0:6, 8288:74848], axis = 0)
    '''
    W_avg4_1=(np.array(W[0][74848:599648])+np.array(W[4][74848:599648])+np.array(W[5][74848:599648]))/3.0
    W_avg4_2=np.mean(W[1:4, 74848:599648], axis = 0)
    
    
    W_avg5_1=(np.array(W[0][599648:615038])+np.array(W[5][599648:615038]))/2.0
    W_avg5_2=np.mean(W[1:4, 599648:615038], axis = 0)
    W_avg5_3=W[4][599648:615038]
    
    W_avg6_1=(np.array(W[0][615038:615224])+np.array(W[5][615038:615224]))/2.0
    W_avg6_2=np.mean(W[1:4, 615038:615224], axis = 0)
    W_avg6_3=W[4][615038:615224]
    '''
   
   
    # print(np.max(W_avg))
    
def reinitialize():
    print(Loss_cache,Loss)
    global iteration_count
    print("The iteration number: ", iteration_count)
    W[W<np.inf] = 0
    Loss[Loss<np.inf] = 1e5
    Loss_cache[Loss_cache<np.inf] = 0
    loss_record[loss_record<np.inf] = 0
    update_flag[update_flag<np.inf] = 1
    normalized_dloss[normalized_dloss<np.inf] = 0
    iteration_count = 0
    global NUM_OF_WAIT
    NUM_OF_WAIT = 9

    global regular
    regular = 1e5
    
    global conver_indicator
    conver_indicator = 1e5
    barrier_update()

    #for i in range(300):
        #print(loss_record[i])

barrier_start = threading.Barrier(NUM_OF_WAIT,action = None, timeout = None)
barrier_W = threading.Barrier(NUM_OF_WAIT,action = server_update, timeout = None)
barrier_end = threading.Barrier(NUM_OF_WAIT, action = reinitialize, timeout = None)

def barrier_update():
    global NUM_OF_WAIT
    print("update the barriers to NUM_OF_WAIT: ",NUM_OF_WAIT)
    global barrier_W
    barrier_W = threading.Barrier(NUM_OF_WAIT,action = server_update, timeout = None)
    global barrier_end
    barrier_end = threading.Barrier(NUM_OF_WAIT, action = reinitialize, timeout = None)

class MyTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        out_i=0
        outer_iter=80
        while out_i<outer_iter:
            try:
                #receive the size of content
                header = self.request.recv(4)
                size = struct.unpack('i', header)
                

                #receive the id of client
                u_id = self.request.recv(4)
                user_id = struct.unpack('i',u_id)
                
                # receive the type of message, defination in communication.py
                mess_type = self.request.recv(4)
                mess_type = struct.unpack('i',mess_type)[0]

                #print("This is the {}th node with message type {}".format(user_id[0],mess_type))

                #receive the body of message
                recv_data = b""
                
                while sys.getsizeof(recv_data)<size[0]:
                    recv_data += self.request.recv(size[0]-sys.getsizeof(recv_data))
                    
                #if hello message, barrier until all clients arrive and send a message to start
                if mess_type == -1:
                    try:
                        barrier_start.wait(1200)
                    except Exception as e:
                        print("start wait timeout...")

                    start_message = 'start'
                    start_data = pickle.dumps(start_message, protocol = 0)
                    size = sys.getsizeof(start_data)
                    header = struct.pack("i",size)
                    self.request.sendall(header)
                    self.request.sendall(start_data)


                #if W message, update Omega and U or F
                elif mess_type == 0:
                    weights = pickle.loads(recv_data)
                    #print(len(weights))
                    #print(out_i, user_id[0])
                    W[user_id[0]-1] = weights

                    try:
                        barrier_W.wait(2400)
                    except Exception as e:
                        print("wait W timeout...")
                        
                    '''
                    if user_id[0]<=4 and user_id[0]>=2:
                        W_avg=np.concatenate((W_avg1_1))#, W_avg2_1,W_avg3_2,W_avg4_2, W_avg5_2, W_avg6_2))
                    elif user_id[0]==5:
                        W_avg=np.concatenate((W_avg1_1))#, W_avg2_1,W_avg3_1,W_avg4_1, W_avg5_3, W_avg6_3))
                    else:
                        W_avg=np.concatenate((W_avg1_1))#, W_avg2_1,W_avg3_1,W_avg4_1, W_avg5_1, W_avg6_1))
                    '''
                                                
                    W_avg=np.concatenate((W_avg1_1, W_avg2_1,W_avg3_1))
                    W_avg_data = pickle.dumps(W_avg, protocol = 0)
                    W_avg_size = sys.getsizeof(W_avg_data)
                    W_avg_header = struct.pack("i",W_avg_size)
                    
                    self.request.sendall(W_avg_header)
                    self.request.sendall(W_avg_data)

                    out_i+=1
                    # print("send Omega to client {} with the size of {}".format(user_id[0],size))


                # if Loss message, record the loss
                elif mess_type == 1:
                    loss = pickle.loads(recv_data)
                    Loss_cache[user_id] = Loss[user_id]
                    Loss[user_id] = (loss + regular)/NUM_OF_TOTAL_USERS

                elif mess_type == 9:
                    break

                elif mess_type == 10:
                    try:
                        barrier_end.wait(5)
                    except Exception as e:
                        print("finish timeout...")
                    break


            except Exception as e:
                print('err',e)
                break



if __name__ == "__main__":
    HOST, PORT = "0.0.0.0", 9999
    server = socketserver.ThreadingTCPServer((HOST,PORT),MyTCPHandler)
    server.serve_forever(poll_interval = 0.5)
