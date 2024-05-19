m=int(input())
n=int(input())
x=[m,n]
x.sort()
seq=[ i+x[0]-1 for i in range(1,x[1]-x[0]+2)]

print(seq) if x==[m,n] else print(seq[::-1])
# 7
# 1
# [7, 6, 5, 4, 3, 2, 1]
# PS C:\Users\Дом\Desktop\2023\lab6_ICT\task2> py .\10.py
# 1
# 7
# [1, 2, 3, 4, 5, 6, 7]