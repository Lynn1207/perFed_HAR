import os
import numpy as np

def concur_simp(acc1, acc2):
    count=0.0
    for i in range(len(acc1)):
        count+=acc1[i]*acc2[i]+(1-acc1[i])*(1-acc2[i])
    return count/float(len(acc1))

accs=[]
meth="FedPerl1"
for i in range(1,13):
    f=open("/home/ubuntu/perFed_HAR/CNN_Human_Activity_Recognition/results/log_com_"+meth+str(i)+".txt")
    for line in f:
        acc=[]
        for i in line:
            if i!=' ' and  i!='\n':
                acc.append(int(i))
    accs.append(acc)
    

#compute the concur_simpleness matrix
concur_m=np.zeros((12,12))
sum_matr=0.0
count=0
for i in range(12):
    for j in range(i+1, 12):
        concur_m[i][j]=concur_simp(accs[i], accs[j])
        count+=1
        sum_matr+=concur_m[i][j]

print(concur_m)

#grouping using concur_simp
avg_consim=sum_matr/count
print("\n Avarage concurrent_simpleness: %.3f. \n"%avg_consim)
for i in range(12):
    for j in range(i+1, 12):
        if concur_m[i][j]>avg_consim:
            concur_m[i][j]=1
        else:
            concur_m[i][j]=0
print(concur_m)


ans=[]
for i in range(12):
    cur={i+1}
    for j in range(i+1, 12):
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
    

           
