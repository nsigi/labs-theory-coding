import copy
import random
import numpy as np

def PrintArr(mes, a):
    print(mes)
    for i in a:
        print(i)

def sumWords(v1, v2):
    v = []
    for i in range(len(v1)):
        v.append((v1[i] + v2[i]) % 2)
    return v

def checkWord(G, H, errors, syndroms):
    word = np.zeros(G.shape[0])
    word[0] = 1
    v = np.dot(word, G) % 2
    print('слово', v)
    e = errors[0]
    #e = errors[0] + [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
    w = sumWords(v, e)
    print("слово с ошибкой: ", w)

    s = np.matmul(w, H) % 2
    print("синдром слова : ", s)

    if tuple(s) in syndroms:
        w = (w + syndroms[tuple(s)]) % 2
        print('исправленное слово: ', w)
        print('проверка: ', np.matmul(w, H) % 2)

def getSindroms(H, errors):
    syndroms = dict()
    for i in range(len(H)):
        syndroms[tuple(np.matmul(errors[i], H))] = errors[i]
    print('syndroms')
    for s in syndroms:
        print(s, syndroms[s])
    return syndroms

def getSindroms2(H, errors):
    syndroms = dict()
    for i in range(n - 1, -1, -1):
        syndroms[tuple(np.matmul(errors[i], H))] = errors[i]

    doubleErrors = []
    for i in range(n):  # генерируем двойные ошибки, комбинируя первые
        for j in range(i + 1, n):
            doubleErrors.append(errors[i] + errors[j])
    for i in range(len(doubleErrors) - 1, -1, -1):
        syndroms[tuple(np.matmul(doubleErrors[i], H) % 2)] = doubleErrors[i]

    return syndroms

def task(G, H):
    errors = np.eye(len(H), dtype=int)
    syndroms = getSindroms(H, errors)
    checkWord(G, H, errors, syndroms)

#_______________________________________________________________________________________________________
print('Часть 1')

r = 3
n = 2 ** r - 1  # Длина кодового слова
k = n - r

X = np.zeros((k, r))

for i in range(r):
    for j in range(k):
        if (j >> i) & 1:  # Проверяем, установлен ли i-й бит в j-м индексе
            X[j, i] = 1
X[0, :] = 1

I1 = np.eye(k, dtype=int)

G = np.concatenate((I1, X), axis=1)
PrintArr('G', G)

H = np.vstack((X, np.eye(n-k)))
PrintArr('H', H)

#task(G, H)

print('_______________________________________________________________________________________________________')
print('Часть 2')

ZRow = np.zeros((1, H.shape[1]))
HZ = np.vstack((H, ZRow))
onesCol = np.ones((HZ.shape[0], 1))

H_ = np.hstack((HZ, onesCol))
sums = [sum(row) for row in G]
b = [1 if s % 2 != 0 else 0 for s in sums]

G_ = np.column_stack((G, b))
PrintArr("G*", G_)
PrintArr('H*', H_)

task(G_, H_)

