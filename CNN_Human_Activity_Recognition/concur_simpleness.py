import os
import numpy as np

def concur_simp(acc1, acc2):
    count=0.0
    for i in range(len(acc1)):
        count+=acc1[i]*acc2[i]+(1-acc1[i])*(1-acc2[i])
    return count/float(len(acc1))

accs=[]
meth="FedAvg"
for i in range(1,7):
    f=open("log_test_"+meth+str(i)+".txt")
    for line in f:
        pass
    acc=[]
    for i in line:
        if i!=' ' and  i!='\n':
            acc.append(int(i))
    accs.append(acc)
    

#compute the concur_simpleness matrix
concur_m=np.zeros((6,6))
for i in range(6):
    for j in range(i+1, 6):
        concur_m[i][j]=concur_simp(accs[i], accs[j])

print(concur_m)
        
        
