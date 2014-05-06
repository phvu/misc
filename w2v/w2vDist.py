import struct, numpy as np

################################################################################
def readString(f):
    s = ''
    c = f.read(1)
    while(c != ' '):
        s += c
        c = f.read(1)
    return s
    
def readFloat4(f):
    bytes = f.read(4)
    assert len(bytes) == 4
    return struct.unpack('f', bytes)[0]

def readFloats(f, n):
    ret = [readFloat4(f) for x in xrange(0, n)]
    assert len(ret) == n
    return ret
################################################################################
 
class Word2VecDistance(object):
    def __init__(self, sFile):
        [self.words, self.features] = self.readDict(sFile)
        
    def readDict(self, sFile):
        with open(sFile, 'rb') as f:
            [nWords, size] = [int(x) for x in f.readline().split()]
            words = [None]*nWords
            features = np.zeros((nWords, size))
            for i in xrange(0, nWords):
                words[i] = readString(f)
                features[i, :] = np.asarray(readFloats(f, size))
                assert f.read(1) == '\n'
        # normalize
        features /= np.linalg.norm(features, axis=1)[:, np.newaxis]
        assert not np.isnan(np.sum(features))
        return (words, features)
            
    def getVector(self, w):
        if w not in self.words:
            return None
        return self.features[self.words.index(w), :].copy()
    
    def getClosestWords(self, v, n):
        assert n > 0 and n + 1 < self.features.shape[0]
        assert v.shape == (self.features.shape[1], )
        
        w = v / np.linalg.norm(v)
        
        # because all vectors are normalized, the cosine distance is 
        # equivalent to the inner product of vectors
        d = (w * self.features).sum(1)
        idx = d.argsort()[::-1]
        topIdx = idx[:n].tolist()
        return ([self.words[i] for i in topIdx], d[topIdx])
        
        
################################################################################

def foo():
    d = Word2VecDistance('/home/hvpham/lib/word2vec/vectors-phrase.bin')
    
    import pickle
    with open('dumpedDistance.dmp', 'wb') as f:
        pickle.dump(d, f)
    with open('dumpedDistance.dmp', 'rb') as f:
        d = pickle.load(f)
   
