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


def getPackError(n, t):
    pack_error = np.zeros(n, dtype=int)
    index_to_put = random.randint(0, n - 1)
    sub_i = 0
    flag = True
    for i in range(t):
        if index_to_put + i == n:
            index_to_put = 0
            sub_i = i
        if flag:
            pack_error[index_to_put + i - sub_i] = 1
        else:
            pack_error[index_to_put + i - sub_i] = random.randint(0, 1)
    print('Ошибка ', pack_error)
    return pack_error


def is_this_err(error, t):
    np.trim_zeros(error, 'f')
    np.trim_zeros(error, 'b')
    return len(error) <= t and len(error) != 0


def coding_with_pack_error(g, n, error_count):
    u = np.zeros(n, dtype=int)
    for i in range(n):
        u[i] = random.randint(0, 1)
    print("Исходное слово", u)
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
                ans = np.array(np.polydiv(sumPoly, g)[0] % 2, dtype=int)
                z = []
                for i in range(len(g) - len(ans)):
                    z.append(0)
                ans = np.concatenate([np.array(z, dtype=int), ans])
                return ans
        else:
            if sum(s_i) <= t:
                e_i = np.zeros(n, dtype=int)
                e_i[i - 1] = 1
                e_x = np.polymul(e_i, s_i)
                e_x %= 2
                sumPoly = np.polyadd(e_x, w)
                sumPoly %= 2
                ans = np.array(np.polydiv(sumPoly, g)[0] % 2, dtype=int)
                z = []
                for i in range(len(g) - len(ans)):
                    z.append(0)
                ans = np.concatenate([np.array(z, dtype=int), ans])
                return ans
    return None


if __name__ == '__main__':
    n = 7
    k = 3
    t = 1
    g1 = np.array([1, 1, 0, 1])
    print("g1", g1)
    print()

    for i in range(1, 4):
        print(i, "ошибки")
        u = coding_with_errors(g1, 4, i)
        decoded = decoding(g1, t, u, False)
        print("Декодированное ", decoded)
        print()

    n2 = 15
    k2 = 9
    t2 = 3
    g2 = np.array([1, 1, 1, 1, 0, 0, 1])
    print("g2", g2)
    print()

    for i in range(1, 5):
        print(i, "ошибки")
        u = coding_with_pack_error(g2, 9, i)
        decoded = decoding(g2, i, u, True)
        print("Декодированное ", decoded)
        print()


