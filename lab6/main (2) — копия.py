
import numpy as np
import random


def coding_with_errors(g, n, error_count):
    u = np.zeros(n, dtype=int)
    for i in range(n):
        u[i] = random.randint(0, 1)
    print("ИСХОДНОЕ", u)
    result = np.polymul(u, g)
    result %= 2

    err_arr = np.zeros(error_count, dtype=int)
    for k in range(error_count):
        mistake_pos = random.randint(0, len(result) - 1)
        while mistake_pos in err_arr:
            mistake_pos = random.randint(0, len(result) - 1)
        err_arr[k] = mistake_pos
        result[mistake_pos] += 1
        result[mistake_pos] %= 2
    return result


def getPackError(n, error_count):
    pack_error = np.zeros(n, dtype=int)
    index_to_put = random.randint(0, n - 1)
    sub_i = 0
    for i in range(error_count):
        if index_to_put + i == n:
            index_to_put = 0
            sub_i = i
        pack_error[index_to_put + i - sub_i] = random.randint(0, 1)
    return pack_error


def is_this_err(error, t):
    np.trim_zeros(error, 'f')
    np.trim_zeros(error, 'b')
    return len(error) <= t and len(error) != 0


def coding_with_pack_error(g, k, error_count):
    u = np.zeros(k, dtype=int)
    for i in range(k):
        u[i] = random.randint(0, 1)
    print("ИСХОДНОЕ", u)
    result = np.polymul(u, g)
    result %= 2
    return result + getPackError(len(result), error_count)


def decoding(g, t, w, isPaket):
    n = len(w)
    s = np.polydiv(w, g)[1]  # остаток
    s %= 2
    for i in range(n):
        e_x = np.zeros(n, dtype=int)
        e_x[n - i - 1] = 1
        mult = np.polymul(s, e_x)
        mult %= 2

        s_i = np.polydiv(mult, g)[1]
        s_i %= 2
        # wt(s_i)
        if isPaket:
            if is_this_err(s_i, t):
                e_i = np.zeros(n, dtype=int)
                e_i[i - 1] = 1
                e_x = np.polymul(e_i, s_i)
                e_x %= 2
                sumPoly = np.polyadd(e_x, w)
                sumPoly %= 2
                return np.polydiv(sumPoly, g)[0] % 2
        else:
            if sum(s_i) <= t:
                e_i = np.zeros(n, dtype=int)
                e_i[i - 1] = 1
                e_x = np.polymul(e_i, s_i)
                e_x %= 2
                sumPoly = np.polyadd(e_x, w)
                sumPoly %= 2
                return np.polydiv(sumPoly, g)[0] % 2
    return None


if __name__ == '__main__':

    '''
    6.1 Написать функции кодирования и декодирования для циклического кода (7,4)
    с порождающим многочленом g(x) = x**3 + x**2 + 1,
    исправляющего однократные ошибки и провести исследование этого кода
    для одно-, двух- и трёхкратных ошибок.
    '''
    print("lab6")
    n = 7
    k = 4
    t = 1
    g1 = np.array([1, 1, 0, 1])
    print("g1", g1)
    print()

    for i in range(1, 4):
        print(i, "mistakes")
        u = coding_with_errors(g1, k, i)
        print("Слово с ошибками", u)
        decoded = decoding(g1, t, u, False)
        print("Декодированное ", decoded)
        print()

    '''
    6.2 Написать функции кодирования и декодирования для циклического кода (15,9)
    с порождающим многочленом g(x) = 1 + x**3 + x**4 + x**5 + x**6,
    исправляющего пакеты ошибок кратности 3 и провести исследование
    этого кода для пакетов ошибок длины 1, 2, 3 и 4.

    Обратите внимание, что пакет ошибок длины t не означает, что все
    разряды в пределах этого пакета изменятся (см. лекции).
    '''
    n2 = 15
    k2 = 9
    t2 = 3
    g2 = np.array([1, 0, 0, 1, 1, 1, 1])
    print("g2", g2)
    print()

    for i in range(1, 5):
        print(i, "mistakes")
        u = coding_with_pack_error(g2, k2, i)
        decoded = decoding(g2, 3, u, True)
        print("Декодированное ", decoded)
        print()