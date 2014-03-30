import numpy as np
import segmentedLsq

def test1():
    d = np.asarray([[1,2,3,4], [3,7,4,2], [7,2,0,-1]])
    (r, x) = segmentedLsq.LeastSquares(d)
    print 'Residuals:', r
    print 'x:', x
    #print 'Computed residuals:', np.linalg.norm(d[-1,:] - np.vstack([d[:-1, :], np.ones(d.shape[1])]).T.dot(x))
    
def test2(C):
    d = np.asarray([[1,2,3,4], [3,7,4,2], [7,2,0,-1]])
    s = segmentedLsq.SegmentedLsq(d)
    (m, b) = s.Segment(C)
    print m, b
    
def test3(C):
    d = np.asarray([[1, 2, 3, 4, 5, 5.1, 5.2, 5.3, 5.4], [1.1, 1.3, 1.3, 1.4, 1.5, 2, 3, 4, 5]])
    s = segmentedLsq.SegmentedLsq(d)
    (m, b) = s.Segment(C)
    print m, b
    
def test4(C):
    d = np.asarray([[1, 2, 3, 4, 5, 5.1, 5.2, 5.3, 5.4, 6, 7, 8, 9, 10], [1.1, 1.3, 1.3, 1.4, 1.5, 2, 3, 4, 5, 6, 5, 4, 3, 2]])
    s = segmentedLsq.SegmentedLsq(d)
    (m, b) = s.Segment(C)
    print m, b
    '''
    Go to fooplot.com and plot those points:
    1,1.1
    2,1.2
    3,1.3
    4,1.4
    5,1.5
    5.1,2
    5.2,3
    5.3,4
    5.4,5
    6,6
    7,5
    8,4
    9,3
    10,2
    '''
