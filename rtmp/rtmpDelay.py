import subprocess, time, sys

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
    
def readHeader(stream):
    data = stream.read(9)
    assert data[:3] == 'FLV'
    assert ord(data[3]) == 0x01
    assert ord(data[4]) == 0x04      # audio
    l = readUInt32BE(data[5:])
    assert l >= 9
    if l > 9:
        data += stream.read(9 - l)
    return data

def readPackage(stream):
    data = stream.read(15)
    l = readUInt24BE(data[5:8])
    data += stream.read(l)
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
        iStart = time.time()
        print 'Sleeping for %f secs... ' % delay,
        sys.stdout.flush()
        while time.time() - iStart < delay:
            allData.append(readPackage(pDumper.stdout))
        print 'Here we go!'
        sys.stdout.flush()
        
        pPlayback = subprocess.Popen(['vlc', '-'], stdin=subprocess.PIPE)
        while 1:
            pPlayback.stdin.write(allData[0])
            allData = allData[1:]
            allData.append(readPackage(pDumper.stdout))
            
    except Exception as e:
        print "Unexpected error:", e
        sys.stdout.flush()
        pDumper.kill()
        pPlayback.kill()

if __name__ == '__main__':
    run(float(sys.argv[1]))
