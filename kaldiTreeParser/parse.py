
class EventMap(object):
    @staticmethod
    def parse(tokens):
        if tokens[-1] == NullEventMap.name():
            return NullEventMap().parse(tokens)
        elif tokens[-1] == ConstantEventMap.name():
            return ConstantEventMap.parse(tokens)
        elif tokens[-1] == SplitEventMap.name():
            return SplitEventMap.parse(tokens)
        elif tokens[-1] == TableEventMap.name():
            return TableEventMap.parse(tokens)
        else:
            print 'Unknown token:', tokens[-1]
            
    def getConstants(self):
        pass
        
class NullEventMap(EventMap):
    @staticmethod
    def name():
        return 'NULL'
    
    @staticmethod
    def parse(tokens):
        assert tokens.pop() == NullEventMap.name()
        return NullEventMap()
    
    def getConstants(self):
        return []
        
class ConstantEventMap(EventMap):
    def __init__(self, pdfId):
        self.pdfId = int(pdfId)
        
    @staticmethod
    def name():
        return 'CE'
        
    @staticmethod
    def parse(tokens):
        assert tokens.pop() == ConstantEventMap.name()
        return ConstantEventMap(tokens.pop())
    
    def getConstants(self):
        return [self.pdfId]
        
class SplitEventMap(EventMap):
    @staticmethod
    def name():
        return 'SE'
        
    @staticmethod
    def parse(tokens):
        assert tokens.pop() == SplitEventMap.name()
        se = SplitEventMap()
        se.key = int(tokens.pop())
        assert tokens.pop() == '['
        se.yesValues = []
        while tokens[-1] != ']':
            se.yesValues.append(int(tokens[-1]))
            tokens.pop()
        assert tokens.pop() == ']'
        assert tokens.pop() == '{'
        se.eventMaps = [EventMap.parse(tokens)]
        se.eventMaps.append(EventMap.parse(tokens))
        assert tokens.pop() == '}'
        return se
    
    def getConstants(self):
        r = []
        for e in self.eventMaps:
            r.extend(e.getConstants())
        return r
                
class TableEventMap(EventMap):
    @staticmethod
    def name():
        return 'TE'
        
    @staticmethod
    def parse(tokens):
        assert tokens.pop() == TableEventMap.name()
        te = TableEventMap()
        te.key = int(tokens.pop())
        sz = int(tokens.pop())
        assert tokens.pop() == '('
        te.eventMaps = [EventMap.parse(tokens) for i in xrange(0, sz)]
        assert tokens.pop() == ')'
        return te
    
    def getConstants(self):
        r = []
        for e in self.eventMaps:
            r.extend(e.getConstants())
        return r
        
################################################################################
   
class ContextDependency(object):
    @staticmethod
    def parse(tokens):
        cd = ContextDependency()
        assert tokens.pop() == 'ContextDependency'
        cd.N = int(tokens.pop())
        cd.P = int(tokens.pop())
        cd.s = tokens.pop()
        cd.eventMap = EventMap.parse(tokens)
        assert tokens.pop() == 'EndContextDependency'
        assert len(tokens) == 0
        return cd
        
################################################################################

def parseContextDependency(sFile):
    tokens = []
    with open(sFile, 'r') as f:
        tokens = ' '.join([s.strip() for s in f.readlines()]).split()
    tokens = tokens[::-1]
    return ContextDependency.parse(tokens)
    
def pdfIdToPhoneMap(cd, bPrint = False):
    e = cd.eventMap
    assert type(e) is TableEventMap
    assert e.key == 1
    print 'There are', len(e.eventMaps), 'entries. Please make sure this is equal to 1 + <number of phones>'
    m = {}
    for i, em in enumerate(e.eventMaps):
        m[i] = em.getConstants()
        if bPrint:
            print i, em.getConstants()
    return m
    
def checkValidMap(m):
    keys = []
    bDup = False
    for k in m:
        for i in m[k]:
            for kk in keys:
                if i in m[kk]:
                    bDup = True
                    print 'Found value', i, 'in the set of IDs for phone', k, 'and', kk
        keys.append(k)

    if not bDup:
        print 'No duplication found (Good!)'
    
    print 'Total number of keys (should be 1 + <number of phones>):', len(m)
    print 'Total number of leaves:', sum([len(m[k]) for k in m])
    print 'Total number of unique pdf-ids:', sum([len(set(m[k])) for k in m])
    totalDupCnt = 0
    for k in m:
        dc = len(m[k]) - len(set(m[k]))
        if dc != 0:
            print 'Found duplicates (%d entries):' % dc, m[k]
        totalDupCnt += dc
    print 'Total number of duplicates:', totalDupCnt

def printDict(d, sFile):
    with open(sFile, 'w') as f:
        f.write('\n'.join([('%d %s' % (k, ' '.join([str(x) for x in set(d[k])]))) for k in d if len(d[k]) > 0]))
        
if __name__ == '__main__':
    import sys
    sFile = 'tree_text'
    if len(sys.argv) > 1:
        sFile = sys.argv[1]
    cd = parseContextDependency(sFile)
    m = pdfIdToPhoneMap(cd)
    checkValidMap(m)
    if len(sys.argv) > 2:
        printDict(m, sys.argv[2])
