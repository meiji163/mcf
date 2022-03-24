import numpy as np

def split(x):
    t = 134217729.0 * x
    s0 = t - (t-x) 
    s1 = x - s0 
    return s0, s1

def two_sum(a, b):
    x = a+b
    bv = x-a
    av = x-bv
    return x, (a-av) + (b-bv)

def two_prod(a, b):
    x = a * b
    a_hi, a_lo = split(a)
    b_hi, b_lo = split(b)
    e1 = x - (a_hi * b_hi)
    e2 = e1 - (a_lo * b_lo)
    e3 = e2 - (a_hi * b_lo)
    y = a_lo * b_lo - e3
    return x, y

def mcf_add_float(a, f):
    b, m = a.shape
    out = np.zeros((b,m+1), dtype=np.float64)
    q = f
    for i in range(m):
        q, out[:,i] = two_sum(q, a[:,i])
    out[:,m] = q
    return out

def mcf_times_float(a, f):
    b, m = a.shape
    e = np.zeros((b,), dtype=np.float64)
    out = np.zeros((b,m+1), dtype=np.float64)
    for i in range(m): 
        t, e1 = two_prod(e[:,i], f)
        out[:,i], e2 = two_sum(t, e)
        e = e1 + e2
    out[:,m] = e

def mcf_add(a, b):
    assert a.shape == b.shape
    batch, m = a.shape
    out = np.zeros((batch,m+1), dtype=np.float64)
    e = np.zeros((batch,), dtype=np.float64)
    for i in range(m):
        t, e1 = two_sum(a[:,i], b[:,i])
        out[:,i], e2 = two_sum(t, e)
    out[:,m] = e
    return out

def renorm(a):
    b = a.shape[0]
    m = a.shape[1] - 1 
    k = np.zeros((b,), dtype=np.int32)
    s = a[:,m]
    tmp = np.zeros_like(a, dtype=np.float64)
    out = np.zeros((b,m), dtype=np.float64)

    for i in reversed(range(1,m+1)):
        s, t[:,i] = two_sum(a[:,i-1], s)

    for i in range(1,m+1):
        s, e = two_sum(s, t[:,i])
        idx = np.nonzero(e)[0]
        out[idx, k[idx]] = s[idx]
        s[idx] = e[idx]
        k += np.where(e!=0., 1, 0)
    return out
