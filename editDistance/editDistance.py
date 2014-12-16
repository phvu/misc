import numpy as np

class EditDistance(object):

    def __init__(self):
        pass

    def costInsert(self, s):
        return 1
    
    def costDelete(self, s):
        return 1
    
    def costSubstitution(self, target, source):
        return (0 if target == source else 2)
        
    def minEditDistance(self, target, source):
        n = len(target)
        m = len(source)
        assert n > 0 and m > 0
        
        distance = np.zeros((n+1, m+1))
        
        for i in xrange(1, n+1):
            distance[i, 0] = distance[i-1, 0] + self.costInsert(target[i-1])
        for j in xrange(1, m+1):
            distance[0, j] = distance[0, j-1] + self.costDelete(source[j-1])
            
        for i in xrange(1, n+1):
            for j in xrange(1, m+1):
                d = [distance[i-1, j] + self.costInsert(target[i-1]), \
                    distance[i-1, j-1] + self.costSubstitution(source[j-1], target[i-1]), \
                    distance[i, j-1] + self.costDelete(source[j-1])]
                distance[i, j] = min(d)
             
        return distance[n, m]

    def minEditDistanceDebug(self, target, source):
        n = len(target)
        m = len(source)
        assert n > 0 and m > 0
        
        distance = np.zeros((n+1, m+1))
        backPt = np.zeros((n+1, m+1), dtype=int)
        opCount = np.zeros((n+1, m+1), dtype=int)
        backPt[0, 1:] = 3
        backPt[1:, 0] = 1
        opCount[0, 1:] = xrange(1, m+1)
        opCount[1:, 0] = xrange(1, n+1)
        
        for i in xrange(1, n+1):
            distance[i, 0] = distance[i-1, 0] + self.costInsert(target[i-1])
        for j in xrange(1, m+1):
            distance[0, j] = distance[0, j-1] + self.costDelete(source[j-1])
            
        for i in xrange(1, n+1):
            for j in xrange(1, m+1):
                d = [distance[i-1, j] + self.costInsert(target[i-1]), \
                    distance[i-1, j-1] + self.costSubstitution(source[j-1], target[i-1]), \
                    distance[i, j-1] + self.costDelete(source[j-1])]
                distance[i, j] = min(d)
                op = [opCount[i-1, j] + 1, \
                    opCount[i-1, j-1] + (0 if source[j-1] == target[i-1] else 1), \
                    opCount[i, j-1] + 1]
                backPt[i, j] = 1 + np.argmin(op)
                opCount[i, j] = min(op)
             
        # backtrace   
        counts = [0, 0, 0]
        alignedStrings = [[], []]
        i = n
        j = m
        
        while i != 0 or j != 0:
            pt = backPt[i, j]
            assert pt in [1, 2, 3]
            if pt == 1:
                counts[pt-1] += 1
                alignedStrings[0].append(target[i-1])
                alignedStrings[1].append('*')
                i -= 1
            elif pt == 2:
                counts[pt-1] += (0 if target[i-1] == source[j-1] else 1)
                alignedStrings[0].append(target[i-1])
                alignedStrings[1].append(source[j-1])
                i -= 1
                j -= 1
            else:
                counts[pt-1] += 1
                alignedStrings[0].append('*')
                alignedStrings[1].append(source[j-1])
                j -= 1
        alignedStrings = [s[::-1] for s in alignedStrings]
        return (counts, distance[n, m], alignedStrings)
        
if __name__ == '__main__':
    import sys
    E = EditDistance()
    (c, d, s) = E.minEditDistanceDebug(sys.argv[1], sys.argv[2])
    print 'Distance=%f,' % d, 'I=%d, S=%d, D=%d' % tuple(c)
    print 'Aligned sequences:', s
