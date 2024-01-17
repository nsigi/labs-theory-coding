import copy
import numpy as np
def ToDecimal(s):
    return int(s.replace(' ','')[:-1], 2)

def REF(arr):
    global gRow
    row = col = 0
    for j in range(len(arr[0])): # по столбцам
        col = j
        if 1 in arr[:, col]:
            for k in range(len(arr)): # ищем ведущую 1
                if arr[k][col] == 1:
                    row = k # запоминаем
                    leads.append([gRow + k, col])
                    break
            break

    buf = arr[0]
    arr[0] = arr[row]
    arr[row] = buf

    for i in range(1, len(arr)):
        if arr[i][col] == 1:
            arr[i] += arr[0]
            arr[i] = arr[i] % 2

    if len(arr) == 1:
        return
    gRow += 1
    REF(arr[1:, :])
    return arr

def RREF(arr):
    for row in range(1, len(arr)):
        col = -1
        for j in range(row, len(arr[0])): # поиск ведущей 1 в i-й строке
            if arr[row][j] == 1:
                col = j
                break
        if (col == -1):
            return arr
        for i in range(0, row): # складываем по модулю 2 все 1-цы выше ведущей
            if arr[i][col] == 1:
                arr[i] += arr[row]
                arr[i] = arr[i] % 2
    return arr

def deleteZeroesStrings(arr):
    newArr = []
    for i in range(len(arr)):
        if ToDecimal("".join([str(j) for j in arr[i]])) != 0:
            newArr.append(arr[i])
    return newArr

'''def SortArr(a):
    d = dict()
    for i in range(len(a)):
        d[i] = ToDecimal("".join([str(i) for i in a[i]]))

    res = []
    sortD = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))
    for i in sortD:
        res.append(a[i])
    return res'''

def PrintSize(n, k):
    print('Rows: ', GetN())
    print('Columns: ', GetK())

def GetSize():
    return GetN(), GetK()

def PrintArr(mes, a):
    print(mes)
    for i in a:
        print(i)

def GetN():
    return len(G)

def GetK():
    return len(G[0])

def GetX(arr):
    X = [[] for _ in range(len(arr))]
    for j in range(len(arr[0])):
        if j not in setLeads:
            for i in range(len(arr)):
                X[i].append(arr[i][j])
    return X

def getI(G_):
    arrI = [[0 for _ in range(len(arr[0]) - len(leads))] for _ in range(len(arr[0]) - len(leads))]
    for i in range(len(arrI)):
        arrI[i][i] = 1
    return arrI

def getH():
    H =[]
    for i in range(len(arr[0])):
        H.append(X.pop(0) if i in setLeads else I.pop(0))
    return H

def getD(v, G):
    d = len(v)
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            curD = 0
            for k in range(len(G[0])):
                if G[i][k] != G[j][k]:
                    curD += 1
            d = min(d, curD)
    return d

#_______________________________________________________________________________________________________

arr = np.array([[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
                [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0 ,1 ,0 ,0 ,1],
                [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
                [1, 0, 1 ,1, 1, 0, 0, 0, 0 ,0 ,0]])
leads = []
gRow = 0
newArr = REF(arr)
PrintArr('Ref', newArr)

G = deleteZeroesStrings(newArr)
PrintArr('G', G)

PrintSize(len(G), len(G[0]))

G_ = RREF(G)
G_ = deleteZeroesStrings(G_)
PrintArr('G*', G_)

setLeads = {el[1] for el in leads}
print('leads = ' + str(setLeads))

X = GetX(G_)
PrintArr('X:', X)

I = getI(G_)
PrintArr('I:', I)

H = getH()
PrintArr('H:', H)

print()
print('1.4.2')

u = np.array([1, 0, 1, 1, 0])
v = np.matmul(u, G) % 2 #u@G = [1 0 1 1 1 0 1 0 0 1 0]
print('v@G = ' + str(v))
v = np.matmul(v, H) % 2 # #v@H = [0 0 0 0 0 0]
print('v@H = ' + str(v))

print()
print(1.4)

PrintArr('G:', G)
print('n = ' + str(GetN()))
print('k = ' + str(GetK()))
print()
v = [1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0]
d = getD(v, G)
print('d = ' + str(d))
print('t = ' + str(d - 1))
print('v = ', v)
e1 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
print('(v + e1) = ', (v + e1) % 2) # = [1 0 0 1 1 0 1 0 0 1 0]
print(np.matmul((v + e1), H) % 2, '– error') #= [0 1 0 0 0 0] – error
e2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
print('(v + e2) = ', (v + e2) % 2)  #= [[1 0 1 1 0 0 1 1 0 1]]
print(np.matmul((v + e2), H) % 2, '– no error')#= [0 0 0 0 0] – no error'''