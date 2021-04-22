import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import math
from statistics import median

user_n=8

def Euc_distance(acc1, acc2):
    count=0.0
    for i in range(len(acc1)):
        count+=(acc1[i]-acc2[i])*(acc1[i]-acc2[i])
    return math.sqrt(count)
    
def concur_simp(acc1, acc2):
    count=0.0
    for i in range(len(acc1)):
        count+=acc1[i]*acc2[i]+(1-acc1[i])*(1-acc2[i])
    return count/float(len(acc1))

accs=[]
meth1="local"
meth=meth1#+"l1"
for i in range(1,user_n+1):
    f=open("/Users/lynn/Documents/MATLAB/federated-multitask-learning-code/logs/log_"+meth1+"/log_com_"+meth+str(i)+".txt")
    for line in f:
        acc=[]
        for i in line:
            if i!=' ' and  i!='\n':
                acc.append(int(i))
    #print(acc)                                        
    accs.append(acc)

'''
paras=[]
for i in range(1,user_n+1):
    para=[]
    f=open("/Users/lynn/Documents/MATLAB/federated-multitask-learning-code/logs/log_"+meth1+"/log_paras"+meth+str(i)+".txt")
    for line in f:
        para.append(float(line))
    paras.append(para)                    

#compute the distance matrix
dist_m=np.zeros((user_n,user_n))
sum_dist=0.0
count=0
for i in range(user_n):
    for j in range(user_n):
        dist_m[i][j]= Euc_distance(paras[i],paras[j])
        if (i!=j):
            count+=1
            sum_dist+=dist_m[i][j]
print(dist_m)

avg_dist=(sum_dist/count)
#show the correlation heatmap
sns.heatmap(dist_m, annot = True, center=avg_dist,cmap= 'coolwarm')
plt.show()
'''
#compute the concur_simpleness matrix
concur_m=np.zeros((user_n,user_n))
sum_matr=0.0
tmp=[]
count=0
for i in range(user_n):
    for j in range(user_n):
        concur_m[i][j]=concur_simp(accs[i], accs[j])
        if (i!=j):
            count+=1
            sum_matr+=concur_m[i][j]
            tmp.append(concur_m[i][j])

print(concur_m)

#grouping using concur_simp
avg_consim=sum_matr/count
print("\n Avarage concurrent_simpleness: %.3f. \n"%avg_consim)

#show the correlation heatmap
sns.heatmap(concur_m, annot = True, center=avg_consim,cmap= 'coolwarm')
plt.show()



for i in range(user_n):
    for j in range(user_n):
        if concur_m[i][j]>=avg_consim:
            concur_m[i][j]=1
        else:
            concur_m[i][j]=0
print(concur_m)



ans=[]
for i in range(user_n):
    cur={i+1}
    for j in range(i+1, user_n):
        if concur_m[i][j]==1:
            cur.add(j+1)
    ans.append(cur)
print(ans)

print("Grouping results: ")
i=0
while i<len(ans):
    no_intersect=False
    while not no_intersect:
        no_intersect=True
        j=i+1
        while j<len(ans):
            if len(ans[i].intersection(ans[j]))>0:
                no_intersect=False
                ans[i].update(ans[j])
                del ans[j]
            else:
                j+=1
    i+=1
    print("%d:"%i, ans[i-1]) 
    


           
