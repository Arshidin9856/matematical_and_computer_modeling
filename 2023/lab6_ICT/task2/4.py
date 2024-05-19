Age=int(input())
if 0<=Age<=150:
    Group = {
        'Childhood' : 13,
        'Youth' : 24,
        'Maturity' : 59,
        'Old age' : 150
    }
    ind=0 
    for i in Group.values():
        if Age>i: ind+=1
    print(list(Group.keys())[ind])
else: print('Are you sure with age?')
# 4
# Childhood
# PS C:\Users\Дом\Desktop\2023\lab6_ICT\task2> py .\4.py
# 50
# Maturity
# PS C:\Users\Дом\Desktop\2023\lab6_ICT\task2> py .\4.py
# 26
# Maturity
# PS C:\Users\Дом\Desktop\2023\lab6_ICT\task2> py .\4.py
# 14
# Youth