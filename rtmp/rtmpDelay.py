import subprocess, time, sys, datetime

def readUIntBE(s):
    l = len(s)
    assert l == 3 or l == 4
    n = 0
    for i in xrange(0, l):
        n |= (ord(s[i]) << (l-i-1))
    return n
    
def readUInt32BE(s):
    assert len(s) == 4
    return readUIntBE(s)

def readUInt24BE(s):
    assert len(s) == 3
    return readUIntBE(s)

def securedRead(stream, n):
    s = ''
    while len(s) < n:
        s += stream.read(n - len(s))
    assert len(s) == n
    return s
    
def readHeader(stream):
    data = securedRead(stream, 9)
    assert data[:3] == 'FLV'
    assert ord(data[3]) == 0x01
    assert ord(data[4]) == 0x04      # audio
    l = readUInt32BE(data[5:])
    assert l >= 9
    if l > 9:
        data += securedRead(stream, 9 - l)
    return data

def readPackage(stream):
    data = securedRead(stream, 15)
    l = readUInt24BE(data[5:8])
    data += securedRead(stream, l)
    return data
    
def run(delay):
    RTMP_PATH = 'rtmpdump'
    RTMP_URL = 'rtmp://210.245.60.242:1935/vov3'
    RTMP_PLAYPATH = 'vov3'
    
    try:
        print 'Opening stream'
        sys.stdout.flush()
        
        pDumper = subprocess.Popen([RTMP_PATH, '-r', RTMP_URL, '--playpath=' + RTMP_PLAYPATH], \
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        allData = []
        allData.append(readHeader(pDumper.stdout))
        
        print 'Sleeping for %f secs... ' % delay,
        sys.stdout.flush()
        
        #iStart = time.time()
        #while time.time() - iStart < delay:
        #    allData.append(readPackage(pDumper.stdout))
        iStart = datetime.datetime.now()
        while (datetime.datetime.now() - iStart).total_seconds() < delay:
            allData.append(readPackage(pDumper.stdout))
            
        print 'Here we go!'
        sys.stdout.flush()
        
        pPlayback = subprocess.Popen(['vlc', '-'], stdin=subprocess.PIPE)
        while 1:
            pPlayback.stdin.write(allData[0])
            allData = allData[1:]
            allData.append(readPackage(pDumper.stdout))
            
    except:
        print "Unexpected error:", sys.exc_info()
        sys.stdout.flush()
        pDumper.kill()
        pPlayback.kill()

if __name__ == '__main__':
    run(float(sys.argv[1]))
