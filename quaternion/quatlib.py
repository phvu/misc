import numpy as np
from numpy.linalg import norm
import scipy.optimize as optimize

def isQuaternion(q):
    '''Check if *q* is a quaternion (a 4-dimensional vector)
    '''
    return type(q) is np.ndarray and q.shape == (4, )

def isRotationQuaterion(q):
    '''Check if *q* is a valid rotation quaternion (a 4-dimensional *unit* vector)
    '''
    return isQuaternion(q) and round(norm(q), 5) == 1.0

###############################################################################

def rot2Quat(theta, axis):
    '''Convert a rotation of *theta* (radians) over *axis* into a quaternion
    :param axis: can be a list, tuple or numpy.ndarray of length 3
    '''
    mAxis = None
    if type(axis) is tuple or type(axis) is list:
        mAxis = np.asarray(axis)
    elif type(axis) is np.ndarray:
        mAxis = axis.copy()
    else:
        raise Exception('axis must be list, tuple or numpy.ndarray')
    assert mAxis.shape == (3, )
    mAxis = mAxis / norm(mAxis)
    mRet = np.asarray([np.cos(theta/2.)] + (np.sin(theta/2.)*mAxis).tolist())
    assert isRotationQuaterion(mRet)
    return mRet

def quat2Rot(q):
    '''Convert a unit quaternion into a rotation.
    Return a tuple (theta, axis), where *theta* is the rotation in radians,
    and *axis* is a 3-dimensional unit vector (of type numpy.ndarray) 
    representing the rotation axis.
    Raise an exception if *q* is not a unit quaternion.
    '''
    assert isRotationQuaterion(q)
    u = q[1:]
    nu = norm(u)
    return (2.*np.arctan2(nu, q[0]), u / nu)

def quat2RotMatrix(q):
    '''Convert a unit quaternion into a 3x3 rotation matrix.
    '''
    assert isRotationQuaterion(q)
    w, x, y, z = q
    m = np.asarray([[w*w + x*x - y*y - z*z, 2*(x*y - w*z), 2*(x*z + w*y)], 
                    [2*(x*y + w*z), w*w - x*x + y*y - z*z, 2*(y*z - w*x)],
                    [2*(x*z - w*y), 2*(y*z + w*x), w*w - x*x - y*y + z*z]])

    return m

###############################################################################

def conjugate(q):
    '''Return the conjugate of quaternion *q*
    '''
    assert isQuaternion(q)
    p = q.copy(); p[1:] = -p[1:]
    return p

def inverse(q):
    '''Return the inverse of quaternion *q*, which is conjugate(q)/norm(q)
    '''
    return conjugate(q) / (norm(q)**2)

def mult(q1, q2):
    '''Quaternion multiplication
    Return q = q1 o q2
    '''
    assert isQuaternion(q1) and isQuaternion(q2)
    w1 = q1[0]; u1 = q1[1:]
    w2 = q2[0]; u2 = q2[1:]
    return np.asarray([w1*w2 - u1.dot(u2)] +
        (w1*u2 + w2*u1 + np.cross(u1, u2)).tolist())

###############################################################################

def rotate(q, x):
    assert isRotationQuaterion(q)
    assert type(x) is np.ndarray and x.shape == (3, )
    x0 = np.asarray([0.] + x.tolist())
    return mult(mult(q, x0), inverse(q))[1:]

###############################################################################

'''

def objectiveFunc(xVal, v, b):
    w, x, y, z = xVal
    return np.asarray([x*z + w*y - v[0]/(2.*b), \
                    y*z - w*x - v[1]/(2.*b), \
                    w*w - x*x - y*y + z*z - v[2]/b, \
                    w*w + x*x + y*y * z*z - 1])

def jacobianFunc(xVal, v, b):
    w, x, y, z = xVal
    return np.asarray([[y, z, w, x], \
                    [-x, -w, z, y], \
                    [2*w, -2*x, -2*y, 2*z], \
                    [2*w, 2*x, 2*y, 2*z]])

def estimate_old(v, r0 = np.asarray([ 0.70710678, 0.40824829, 0.40824829, 0.40824829])):
    ''Estimate the rotation quaternion needed in order to rotate a gravity vector
    obtained from phone's accelerometer to make it aligned to the canonical 
    gravity vector of (0, 0, 9.81)
    :param v: the gravity vector obtained from phone,
    should be acceleration_with_gravity - acceleration_without_gravity
    ''
    assert type(v) is np.ndarray and v.shape == (3, )
    vnorm = np.linalg.norm(v)
    r = optimize.fsolve(objectiveFunc, r0, args=(v, vnorm), fprime=jacobianFunc)
    r /= np.linalg.norm(r)
    r[0] = -r[0]
    return r
'''

def estimate(v1, v2):
    '''Estimate the rotation quaternion needed in order to rotate vector v1 into v2.
    It is assumed that v1 and v2 has the same length (i.e. their Euclidean norms are equal),
    otherwise the rotation quaternion can not be determined correctly.
    '''
    assert type(v1) is np.ndarray and v1.shape == (3, )
    assert type(v2) is np.ndarray and v2.shape == (3, )
    assert round(norm(v1), 5) == round(norm(v2), 5), 'v1 and v2 must have the same length'

    n = np.cross(v1, v2)
    n /= norm(n)
    xHalf = np.arccos(v1.dot(v2)/(norm(v1)*norm(v2)))*0.5
    sinxHalf = np.sin(xHalf)
    return np.asarray([np.cos(xHalf), sinxHalf*n[0], sinxHalf*n[1], sinxHalf*n[2]])