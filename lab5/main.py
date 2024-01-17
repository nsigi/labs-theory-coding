import itertools
from operator import itemgetter
import numpy as np
from itertools import product, combinations
import math


def ToDecimal(s):
    s = str(s)
    s = s[1:]
    return int(s.replace(' ', '')[:-1], 2)


# генерируем бинарную матрицу размерности m столбцов
def get_base(m):
    b = list(product([0, 1], repeat=m))
    b = np.array([x[::-1] for x in b])
    return b


def get_combs(m):
    res = []
    for i in range(r + 1):
        res.extend(list(itertools.combinations(range(m - 1, -1, -1), i)))
    return res


def get_f_I(x, I):
    f = 1
    for i in I:
        f *= (x[i] + 1) % 2
    return f


def get_v_I(I, m):
    return np.ones(2 ** m, int) if len(I) == 0 \
        else [get_f_I(x, I) for x in get_base(m)]


def get_size(r, m):
    size = 0
    for i in range(r + 1):
        size += math.comb(m, i)
    return size


def get_combs_major(r, m):
    combs = [i for i in range(m)]
    res = list(combinations(combs, r))
    if r != 0:
        res.sort(key=itemgetter(r - 1))

    return res


def RM(r, m):
    G = np.zeros((get_size(r, m), pow(2, m)), dtype=int)
    pairs = get_combs(m)
    combs = get_base(m)
    for i in range(len(pairs)):
        pair = pairs[i]
        for j in range(len(combs)):
            comb = combs[j]
            is_valid = True
            for k in range(len(pair)):
                if (comb[pair[k]] == 1):
                    is_valid = False
            if is_valid:
                G[i, j] = 1

    return G

def sumWords(v1, v2):
    v = []
    for i in range(len(v1)):
        v.append((v1[i] + v2[i]) % 2)
    return np.array(v)


# комплиментарное множество (эл-ты которые не входят в I[i])
def get_komp(I, m):
    komplement = []
    for i in range(m):
        if i not in I:
            komplement.append(i)
    return komplement


# добавляем слова, где f_I = 1
def H_I(I, m):
    H = []
    for w in get_base(m):
        if get_f_I(w, I) == 1:
            H.append(w)
    return H


# ищем f_I_t
def get_f_I_t(x, I, t):
    f = 1
    for i in I:
        f *= (x[i] + t[i] + 1) % 2
    return f


# ищем V_I_t
def get_V_I_t(I, m, t):
    return np.ones(2 ** m, int) if len(I) == 0 \
        else [get_f_I_t(x, I, t) for x in get_base(m)]


def major_decode(word, r, m, size):
    i = r
    w_i = word
    m_j = np.zeros(size, dtype=int)
    max_weight = pow(2, m - r - 1) - 1
    index = 0
    while True:
        for J in get_combs_major(i, m):
            max_count = pow(2, m - i - 1)
            counter = [0, 0]
            for t in H_I(J, m):
                c = np.dot(w_i, get_V_I_t(get_komp(J, m), m, t)) % 2
                counter[c] += 1

            if counter[0] > max_weight and counter[1] > max_weight:
                return None
            if counter[0] > max_count:
                m_j[index] = 0
                index += 1
            if counter[1] > max_count:
                m_j[index] = 1
                index += 1
                w_i = sumWords(w_i, get_v_I(J, m))

        if i > 0:
            if len(w_i) <= max_weight:
                for _ in get_combs_major(r + 1, m):
                    m_j[index] = 0
                    index += 1
                break
            i -= 1
        else:
            break

    return m_j[::-1]


def task(G, r, m, countError):
    word = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1], dtype=int)
    print("Начальное слово:\n", word)
    v = word.dot(G) % 2
    print("Кодовое слово:\n", v)
    e = np.zeros(len(v), dtype=int)
    for i in range(countError):
        e[i] = 1
    w = sumWords(v, e)
    print(f"Слово с ошибкой кратности {countError}:\n", w)

    corr_w = major_decode(w, r, m, len(G))
    if corr_w is None:
        print("Необходима повторно отправить сообщение")
    else:
        print("Исправленное слово: \n", corr_w)
        print("Проверка кодового слова: \n", corr_w.dot(G) % 2)


def PrintArr(mes, a):
    print(mes)
    for i in a:
        print(i)
    print()


r = 2
m = 4
n = 2 ** m
G = RM(r, m)
PrintArr('G', G)

task(G, r, m, 2)
