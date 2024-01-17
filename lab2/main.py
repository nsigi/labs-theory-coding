import copy
import random

import numpy as np
def ToDecimal(s):
    return int(s.replace(' ','')[:-1], 2)

'''def REF(arr):
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
    return arr'''

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

'''def getI(G_):
    arrI = [[0 for _ in range(len(arr[0]) - len(leads))] for _ in range(len(arr[0]) - len(leads))]
    for i in range(len(arrI)):
        arrI[i][i] = 1
    return arrI'''

def getH(X, I):
    H =[]
    for i in range(len(G[0])):
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


def correctWord(H, sindrom, word):
    i = -1
    for row in range(0, len(H)):
        counter = 0
        for j in range(0, len(H[0])):
            if sindrom[j] == H[row][j]:
                counter += 1
        if counter == len(H[0]):
            i = row

    if i == -1:
        print("Такого синдрома нет в Н")
        return False
    word[i] = (word[i] + 1) % 2
    return True

#_______________________________________________________________________________________________________
'''print('Часть 1')

n, k, d = 7, 4, 3
X = [ [0, 1, 1],
      [1, 0, 1],
      [1, 1, 1],
      [1, 1, 0]]

I1 = np.eye(k, dtype=int)

G = np.concatenate((I1, X), axis=1)
PrintArr('G', G)

I2 = [[1, 0, 0],
      [0, 1, 0],
      [0, 0, 1]]
setLeads = {0, 1, 2, 3}
H = getH(X, I2)
PrintArr('H:', H)

errors = np.eye(7, dtype=int)
syndroms = dict()
for i in range(n - 1, -1, -1):
    syndroms[tuple(np.matmul(errors[i], H))] = errors[i]
print('syndroms')
for s in syndroms:
    print(s, syndroms[s])

v = [1, 0, 0, 0, 1, 1, 1]
print('слово', v)
print()
e1 = [0, 0, 0, 0, 0, 0, 1]
w1 = sumWords(v, e1)
print("слово с одной ошибкой: ", w1)

s1 = np.matmul(w1, H) % 2
print("синдром слова : ", s1)

if correctWord(H, s1, w1):
    print('исправленное слово: ', w1)
    print('проверка: ', np.matmul(w1, H) % 2)
    
print()
e2 = [1, 0, 1, 0, 0, 0, 0]
w2 = sumWords(v, e2)
print("слово с двумя ошибками: ", w2)

s2 = np.matmul(w2, H) % 2
print("синдром слова : ", s2)

if correctWord(H, s2, w2):
    print('исправленное слово: ', w2)
    print('проверка: ', np.matmul(w2, H) % 2)
print('____________________________________________________________')'''
def sumWords(v1, v2):
    v = []
    for i in range(len(v1)):
        v.append((v1[i] + v2[i]) % 2)
    return v

def generateX(n ,k):
    flag = True
    X = []
    while flag:
        flag = False
        X = [[1 if random.random() > 0.5 else 0 for _ in range(n)] for _ in range(k)]

        for i in range(k): # не менее 4 едениц
            if sum(X[i]) < 4:
                flag = True
                continue

        for i in range(k - 1): # сумма любых двух строк содержала не менее 3 единиц;
            for j in range(i + 1, k):
                if sum(sumWords(X[i], X[j])) < 3:
                    flag = True
                    continue

        for i in range(k - 2): # сумма любых трёх строк содержала не менее 2 единиц;
            for j in range(i + 1, k - 1):
                for m in range(j + 1, k):
                    s = sumWords(X[i], X[j])
                    s = sumWords(s, X[m])
                    if sum(s) < 2:
                        flag = True

        for i in range(k - 3): #сумма любых четырёх строк содержала не менее 1 единицы;
            for j in range(i + 1, k - 2):
                for m in range(j + 1, k - 1):
                    for p in range(m + 1, k):
                        s = sumWords(X[i], X[j])
                        s = sumWords(s, X[m])
                        s = sumWords(s, X[p])
                        if sum(s) < 1:
                            flag = True
    return X

print('Часть 2')
n, k, d = 11, 4, 5
setLeads = {0, 1, 2, 3}
X = generateX(n - k, k)

'''X = [[1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 1, 0, 1, 0],
[0, 1, 0, 1, 1, 1, 1],
[1, 0, 1, 0, 1, 1, 0]]'''

I1 = np.eye(k, dtype=int)

I2 = [[1, 0, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 0, 1]]

G = np.concatenate((I1, X), axis=1)
PrintArr('G', G)

H = getH(X, I2)
PrintArr('H:', H)

errors = np.eye(n, dtype=int)
syndroms = dict()
for i in range(n - 1, -1, -1):
    syndroms[tuple(np.matmul(errors[i], H))] = errors[i]

doubleErrors = []
for i in range(n): # генерируем двойные ошибки, комбинируя первые
    for j in range(i + 1, n):
        doubleErrors.append(errors[i] + errors[j])
for i in range(len(doubleErrors) - 1, -1, -1):
    syndroms[tuple(np.matmul(doubleErrors[i], H) % 2)] = doubleErrors[i]

print('syndroms')
for s in syndroms:
    print(s, syndroms[s])
print()

u = [1, 0, 0, 0]
v = np.matmul(u, G) % 2
print('слово', v)
print()
#e1 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
#e1 = [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
e1 = [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
w1 = sumWords(v, e1)
print("слово с одной ошибкой: ", w1)

s1 = np.matmul(w1, H) % 2
print("синдром слова : ", s1)

print('Yes' if tuple(s1) in syndroms else 'NO')
if tuple(s1) in syndroms:
    w1 = (w1 + syndroms[tuple(s1)]) % 2
    print('исправленное слово: ', w1)
    print('проверка: ', np.matmul(w1, H) % 2)

#print(len(syndroms))
