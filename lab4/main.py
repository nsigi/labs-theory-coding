import copy
import random
import numpy as np

H_kron = [[1, 1],
          [1, -1]]


def getH(i, m):
    return mult_kron(mult_kron(np.eye(2 ** (m - i), dtype=int), H_kron), np.eye(2 ** (i - 1), dtype=int))

def mult_kron(M1, M2):
    K = copy.copy(M2)
    for i in range(len(M1[0]) - 1):
        K = np.hstack((K, M2))
    line = copy.copy(K)
    for i in range(len(M1[0]) - 1):
        K = np.vstack((K, line))
    k = [[K[i][j] * M1[i // len(M2)][j // len(M2[0])] for j in range(len(K[0]))] for i in range(len(K))]

    return k


def getB(k): # формируется сдвигом влево
    B = np.zeros((k, k), dtype=int)
    B[0] = [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1]
    for i in range(k - 1):
        B[i + 1] = np.roll(B[i], -1)
    return B


def RM(r, m):
    if 0 < r < m:
        G11 = RM(r, m - 1)
        G22 = RM(r - 1, m - 1)
        G_left = np.vstack((G11, np.zeros((G22.shape[0], G11.shape[1]), dtype=int)))
        G_right = np.vstack((G11, G22))
        return np.hstack((G_left, G_right))
    elif r == 0:
        return np.ones((1, 2 ** m), dtype=int)
    elif r == m:
        G_top = RM(r - 1, m)
        G_bot = np.zeros((1, 2 ** m), dtype=int)
        G_bot[0][-1] = 1
        return np.vstack((G_top, G_bot))


def PrintArr(mes, a):
    print(mes)
    for i in a:
        print(i)


def sumWords(v1, v2):
    v = []
    for i in range(len(v1)):
        v.append((v1[i] + v2[i]) % 2)
    return v


def checkWord1(G, H, errors, syndroms, errorCount):
    word = np.zeros(G.shape[0], dtype=int)
    word[0] = 1
    print('исходное слово', word)
    v = np.dot(word, G) % 2
    print('закодированное слово', v)
    e = errors[0]
    for i in range(1, errorCount):
        e += errors[-i]
    w = sumWords(v, e)
    print('слово с ошибкой', w)

    s = np.matmul(w, H) % 2
    print("синдром слова : ", s)

    if tuple(s) in syndroms:
        w = (w + syndroms[tuple(s)]) % 2
        print('исправленное слово: ', w)
        print('проверка: ', np.matmul(w, H) % 2)
    else:
        print('нельзя исправить')

def getSindroms(H, errors):
    syndroms = dict()
    for i in range(len(H)):
        syndroms[tuple(np.matmul(errors[i], H))] = errors[i]

    return syndroms

def getSindroms2(H, errors):
    syndroms = getSindroms(H, errors)

    doubleErrors = []
    for i in range(len(errors)):  # генерируем двойные ошибки, комбинируя первые
        for j in range(i + 1, len(errors)):
            doubleErrors.append(errors[i] + errors[j])
    for i in range(len(doubleErrors) - 1, -1, -1):
        syndroms[tuple(np.matmul(doubleErrors[i], H) % 2)] = doubleErrors[i]

    return syndroms, doubleErrors

def getSindroms3(H, errors):
    syndroms, doubleErrors = getSindroms2(H, errors)

    tripleErrors = []
    for i in range(len(doubleErrors)):  # генерируем тройные ошибки, комбинируя первые
        for j in range(i + 1, len(errors)):
            tripleErrors.append(doubleErrors[i] + errors[j])
    for i in range(len(tripleErrors) - 1, -1, -1):
        syndroms[tuple(np.matmul(tripleErrors[i], H) % 2)] = tripleErrors[i]

    return syndroms, tripleErrors

def task1(G, H):
    errors = np.eye(len(H), dtype=int)
    #syndroms = getSindroms(H, errors)
    syndroms, dE = getSindroms2(H, errors)
    #syndroms, tE = getSindroms3(H, errors)
    checkWord1(G, H, errors, syndroms, )
    print(len(syndroms))  # n (n + 1) / 2

def checkWord2(G, errors, errorCount, m):
    word = np.zeros(G.shape[0], dtype=int)
    word[0] = word[2] = 1
    print('исходное слово', word)
    v = np.dot(word, G) % 2
    print('закодированное слово', v)
    e = errors[0]
    for i in range(1, errorCount):
        e += errors[-i]
    w = sumWords(v, e)
    print('слово с ошибкой', w)
    # замена 0 на -1
    w = [-1 if i == 0 else 1 for i in w]
    #w = [1, -1, 1, -1, 1, -1, 1, 1]
    wNew = w # s
    for i in range(1, m + 1):
        wNew = np.dot(wNew, getH(i, m))
    maxI = np.where(wNew == max(wNew))[0][0]
    binNum = [int(i) for i in np.binary_repr(maxI, width=m)[::-1]]
    orig = np.zeros(m + 1, dtype=int)
    orig[1:m + 1] = binNum
    if wNew[maxI] > 0:
        orig[0] = 1
    return orig

def task2(G, m):
    errors = np.eye(2 ** m, dtype=int)
    print('исправленное слово', checkWord2(G, errors, 4, m))

# _______________________________________________________________________________________________________
'''print('Часть 1')

n = 24
k = 12
d = 8

B = getB(k)
I = np.eye(B.shape[0], dtype=int)

G = np.concatenate((I, B), axis=1)
PrintArr('G', G)

H = np.vstack((B, np.eye(B.shape[1], dtype=int)))
PrintArr('H', H)

task1(G, H)'''

print('_______________________________________________________________________________________________________')
print('Часть 2')
#r, m = 1, 3
r, m = 1, 4

G = RM(r, m)
PrintArr("G", G)
task2(G, m)
