import socketserver
import pickle, struct
import sys
from threading import Lock, Thread
import threading
import numpy as np
from sklearn.decomposition import PCA
#from keras.utils import to_categorical
import tensorflow.compat.v1 as tf
import math


tf.disable_v2_behavior()
# np.set_printoptions(threshold=np.inf)

NUM_OF_TOTAL_USERS = 8
NUM_OF_WAIT = NUM_OF_TOTAL_USERS
W_DIM =75750#l1: 1664; l2: 52896; l3: 163872, l4: 213152; l5:776806
inner_iteration = 5
T_thresh = 10

iteration_count = 0
regular = 1e5
alpha = 1*(1e-3)

W = np.zeros((NUM_OF_TOTAL_USERS,W_DIM))
W_avg = np.zeros(W_DIM)
#W_update=np.zeros((NUM_OF_TOTAL_USERS,W_DIM))
Loss = np.zeros(NUM_OF_TOTAL_USERS)
Loss[Loss<np.inf] = 1e5
Loss_cache = np.zeros(NUM_OF_TOTAL_USERS)
conver_indicator = 1e5
loss_record = np.zeros(1100)
normalized_dloss = np.zeros((NUM_OF_TOTAL_USERS,T_thresh))
update_flag = np.ones(NUM_OF_TOTAL_USERS)

groups_l1=[{2: 0.171, 3: 0.171, 8: 0.171, 1: 0.142, 5: 0.142, 6: 0.142, 4: 0.057},{7: 1.0}]#[{1:1.0},{2:1.0},{3:1.0},{4:1.0},{5:1.0},{6:1.0},{7:1.0},{8:1.0}]#
W_l1=np.zeros((len(groups_l1),1664))

groups_l2=[{1:1.0},{2: 0.2, 3: 0.2, 5: 0.2, 6: 0.2, 8: 0.2},{7:1.0},{4:1.0}]
W_l2=np.zeros((len(groups_l2),75424-1664))

groups_l3=[{1:1.0},{2: 1.0}, {3: 0.3, 6: 0.3, 8: 0.2, 5: 0.2},{7:1.0},{4:1.0}]
W_l3=np.zeros((len(groups_l3),130912-75424))


def server_update():
    
    global W,W_l1,W_l2,W_l3, W_avg
    # print(np.max(W))
    W_avg=np.mean(W, axis = 0)
    '''
    if W[0][1663]!=0:
        #print("Layer 1")
        for i in range(len(groups_l1)):
            tmp_w=np.zeros(1664)
            for key in groups_l1[i]:
                tmp_w+=groups_l1[i][key]*W[key-1, 0:1664]
            W_l1[i]=tmp_w
    if W[0][75423]!=0:
        #print("Layer 2")
        for i in range(len(groups_l2)):
            tmp_w=np.zeros(75424-1664)
            for key in groups_l2[i]:
                tmp_w+=groups_l2[i][key]*W[key-1, 1664:75424]
            W_l2[i]=tmp_w
    if W[0][130911]!=0:
        #print("Layer 3")
        for i in range(len(groups_l3)):
            tmp_w=np.zeros(130912-75424)
            for key in groups_l3[i]:
                tmp_w+=groups_l3[i][key]*W[key-1, 75424:130912]
            W_l3[i]=tmp_w
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
                    W[user_id[0]-1, 0:len(weights)] = weights

                    try:
                        barrier_W.wait(4800)
                    except Exception as e:
                        print("wait barrier W timeout...", str(barrier_W.n_waiting), e)
                    '''
                    if W[0][1663]!=0:
                        g_i=0
                        for group in groups_l1:
                            if user_id[0] in group:
                                mu=min(group[user_id[0]]*len(group),1.0)
                                W_gen=W_l1[g_i]*mu+(1-mu)*W[user_id[0]-1, 0:1664]
                                if user_id[0]==1:
                                    print(user_id[0],"Layer_1: ", g_i, mu)
                                break
                            g_i+=1
                        
                    if W[0][75423]!=0:    
                        g_i=0
                        for group in groups_l2:
                            if user_id[0] in group: 
                                mu=min(group[user_id[0]]*len(group),1.0)
                                W_gen=np.concatenate((W_gen, W_l2[g_i]*mu+(1-mu)*W[user_id[0]-1, 1664:75424]))
                                if user_id[0]==1:
                                    print(user_id[0],"Layer_2: ", g_i, mu)
                                break
                            g_i+=1
                        
                    if W[0][130911]!=0:
                        g_i=0
                        for group in groups_l3:
                            if user_id[0] in group: 
                                mu=min(group[user_id[0]]*len(group),1.0)
                                W_gen=np.concatenate((W_gen, W_l3[g_i]*mu+(1-mu)*W[user_id[0]-1, 75424:130912]))
                                if user_id[0]==1:
                                    print(user_id[0],"Layer_3: ", g_i,mu)
                                break
                            g_i+=1
                    '''
                    W_gen=W_avg#0.5*W_avg+0.5*W[user_id[0]-1]
                    #print(user_id[0], W_avg.shape)
                    
                    W_avg_data = pickle.dumps(W_gen, protocol = 0)
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
