import numpy as np, scipy.optimize as optimize

def objectiveFunc(xVal, b, a):
    '''b is the source (pre-rotation) coordinate
    a is the destination (post-rotation) coordinate, 
    '''
    w, x, y, z = xVal
    ww, xx, yy, zz = xVal**2
    return np.asarray([ \
        b[0]*(ww + xx - yy - zz) + 2*b[1]*(x*y - w*z) + 2*b[2]*(x*z + w*y) - a[0], \
        b[1]*(ww - xx + yy - zz) + 2*b[0]*(x*y + w*z) + 2*b[2]*(y*z - w*x) - a[1], \
        b[2]*(ww - xx - yy + zz) + 2*b[0]*(x*z - w*y) + 2*b[1]*(w*x + y*z) - a[2], \
        w*w + x*x + y*y * z*z - 1])

def jacobianFunc(xVal, b, a):
    w, x, y, z = xVal
    return np.asarray([ \
        [   2*b[0]*w - 2*b[1]*z + 2*b[2]*y, \
            2*b[0]*x + 2*b[1]*y + 2*b[2]*z, \
           -2*b[0]*y + 2*b[1]*x + 2*b[2]*w, \
           -2*b[0]*z - 2*b[1]*w + 2*b[2]*x], \
        [   2*b[1]*w + 2*b[0]*z - 2*b[2]*x, \
           -2*b[1]*x + 2*b[0]*y - 2*b[2]*w, \
            2*b[1]*y + 2*b[0]*x + 2*b[2]*z, \
           -2*b[1]*z + 2*b[0]*w + 2*b[2]*y], \
        [   2*b[2]*w - 2*b[0]*y + 2*b[1]*x, \
           -2*b[2]*x + 2*b[0]*z + 2*b[1]*w, \
           -2*b[2]*y - 2*b[0]*w + 2*b[1]*z, \
            2*b[2]*z + 2*b[0]*x + 2*b[1]*y], \
        [2*w, 2*x, 2*y, 2*z]])

def estimate(source, dest, r0 = np.asarray([0, 1, 1, 1])):
    '''Estimate the rotation quaternion needed in order to rotate *source* into *dest*
    '''
    assert type(source) is np.ndarray and source.shape == (3, )
    assert type(dest) is np.ndarray and dest.shape == (3, )
    #r, infodict, ier, mesg = optimize.fsolve(objectiveFunc, r0, args=(source, dest), fprime=jacobianFunc, full_output=True)
    sol = optimize.root(objectiveFunc, r0, args=(source, dest), jac=jacobianFunc, method='anderson')
    r = sol.x
    r /= np.linalg.norm(r)
    #return r, infodict, ier, mesg
    return r