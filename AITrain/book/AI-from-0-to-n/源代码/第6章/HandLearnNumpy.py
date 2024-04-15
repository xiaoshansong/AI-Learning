

import numpy as np

matrixa = np.eye(3,k=1)
print(matrixa)

# 索引和切片
arangea = np.arange(0,11)
print(arangea)

print(arangea[0])
print("[2:]" ,arangea[2:])
print("[:2]" , arangea[:2])
print("[-1]" , arangea[-1])
print(arangea[4])
print("[::-1]" , arangea[::-1])
print("[::2]",arangea[::2])

f = lambda m,n:n+10*m
A = np.fromfunction(f,(6,6),dtype=int)
print(A)


data1 = np.array([[1,2],[3,4]])
print(np.reshape(data1,(1,2,2)))
