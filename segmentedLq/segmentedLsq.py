import numpy as np

def LeastSquares(data):
    # quite hacky here, using the last dimension of data as "y"
    A = np.vstack([data[:-1, :], np.ones(data.shape[1])]).T
    y = data[-1, :]
    results = np.linalg.lstsq(A, y)
    x = results[0]
    r = results[1]
    if r.size == 0:
        r = y - A.dot(x)
        assert len(r.shape) == 1 and r.shape[0] == data.shape[1]
        r = np.linalg.norm(r)**2
    else:
        r = r[0]
    return (r, x)

def computeDistanceLsq(data):
    n = data.shape[1]
    e = np.zeros((n, n))
    for j in xrange(0, n):
        for i in xrange(0, j+1):
            e[i, j] = LeastSquares(data[:, i:j+1])[0]
    return e

def computeDistanceAverageEuclidean(data):
    n = data.shape[1]
    eucDistance = np.zeros((n, n))
    for j in xrange(0, n):
        for i in xrange(0, j+1):
            eucDistance[i, j] = np.linalg.norm(data[:, i] - data[:, j])

    e = np.zeros((n, n))
    for j in xrange(0, n):
        for i in xrange(0, j+1):
            e[i, j] = np.sum([eucDistance[i:k+1,k].sum() for k in xrange(i,j+1)])/(j-i+1)
    return e
            
class SegmentedLsq(object):

    def __init__(self, data):
        assert len(data.shape) == 2
        self.data = data
        self.distanceMode = 0
        self.eLsq = None
        self.eAvgEuc = None
        self.computeDistance()
        
    def computeDistance(self):
        if self.distanceMode == 0:
            if self.eLsq is None:
                self.eLsq = computeDistanceLsq(self.data)
            self.e = self.eLsq
        else:
            if self.eAvgEuc is None:
                self.eAvgEuc = computeDistanceAverageEuclidean(self.data)
            self.e = self.eAvgEuc
        
    def Segment(self, C):
        n = self.data.shape[1]
        M = np.zeros((n, 1))
        startIdx = np.zeros((n, 1), dtype=int)
        for j in xrange(1, n):
            v = self.e[:j, j] + C + M[:j, 0]
            startIdx[j, 0] = np.argmin(v)
            M[j, 0] = v[startIdx[j, 0]]
        
        # backtrace
        boundaries = [n]
        idx = n-1
        while idx >= 0:
            boundaries.append(startIdx[idx, 0])
            idx = startIdx[idx, 0]-1
        return (M[-1, 0], boundaries[::-1])
