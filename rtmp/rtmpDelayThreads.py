import subprocess, time, sys, threading, Queue

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

#########################################################################

RTMP_PATH = 'rtmpdump'
RTMP_URL = 'rtmp://210.245.60.242:1935/vov3'
RTMP_PLAYPATH = 'vov3'
exitFlag = 0

class DumperThread (threading.Thread):
    def __init__(self, threadID, name, q, qLock):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.qData = q
        self.qLock = qLock
        
    def run(self):
        pDumper = subprocess.Popen([RTMP_PATH, '-r', RTMP_URL, '--playpath=' + RTMP_PLAYPATH], \
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        sData = readHeader(pDumper.stdout)
        self.qLock.acquire()
        self.qData.put(sData)
        self.qLock.release()

        while not exitFlag:
            sData = readPackage(pDumper.stdout)
            self.qLock.acquire()
            self.qData.put(sData)
            self.qLock.release()
        pDumper.kill()
        
def run(delay):
    try:
        global exitFlag
        print 'Opening stream'
        sys.stdout.flush()
        
        dataQueue = Queue.Queue()
        queueLock = threading.Lock()
        dumperThread = DumperThread(0, 'dumper', dataQueue, queueLock)
        dumperThread.start()

        print 'Sleeping for %f secs... ' % delay,
        sys.stdout.flush()
        time.sleep(delay)
        print 'Here we go!'
        sys.stdout.flush()
        
        pPlayback = subprocess.Popen(['vlc', '-'], stdin=subprocess.PIPE)
        while not exitFlag:
            queueLock.acquire()
            if not dataQueue.empty():
                sData = dataQueue.get()
                queueLock.release()
                pPlayback.stdin.write(sData)
            else:
                queueLock.release()
                time.sleep(0.1)
    except:
        print "Unexpected error:", sys.exc_info()[0]
        sys.stdout.flush()
        exitFlag = 1
        dumperThread.join()
        pPlayback.kill()
        
if __name__ == '__main__':
    run(float(sys.argv[1]))
