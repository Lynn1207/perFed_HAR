import os
import numpy as np

def concur_simp(acc1, acc2):
    count=0.0
    for i in range(len(acc1)):
        count+=acc1[i]*acc2[i]+(1-acc1[i])*(1-acc2[i])
    return count/float(len(acc1))

accs=[]
meth="FedPerl5"
for i in range(1,7):
    f=open("/home/ubuntu/perFed_HAR/CNN_Human_Activity_Recognition/results/log_com_"+meth+str(i)+".txt")
    for line in f:
        acc=[]
        for i in line:
            if i!=' ' and  i!='\n':
                acc.append(int(i))
    accs.append(acc)
    

#compute the concur_simpleness matrix
concur_m=np.zeros((6,6))
sum_matr=0.0
count=0
for i in range(6):
    for j in range(i+1, 6):
        concur_m[i][j]=concur_simp(accs[i], accs[j])
        count+=1
        sum_matr+=concur_m[i][j]

print(concur_m)

#grouping using concur_simp
avg_consim=sum_matr/count
print("\n Avarage concurrent_simpleness: %.3f. \n"%avg_consim)
for i in range(6):
    for j in range(i+1, 6):
        if concur_m[i][j]>avg_consim:
            concur_m[i][j]=1
        else:
            concur_m[i][j]=0

print(concur_m)
