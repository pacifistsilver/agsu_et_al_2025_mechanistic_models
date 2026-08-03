def tbound(a_s, b_s, a_n, b_n):
    numerator = (a_s/b_s) + (a_n/b_n)
    denom = a_s + a_n
    return numerator / denom
