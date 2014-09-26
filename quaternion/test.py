import quatlib as ql
import numpy as np

def test1():
    q = ql.rot2Quat(np.pi/2, [0, 1, 0])
    v = np.asarray([1, 2, 3])
    a = ql.rotate(q, v)
    m = ql.quat2RotMatrix(q)
    b = m.dot(v)
    assert np.all(a - b <= 1E-10)


if __name__ == '__main__':
    test1()