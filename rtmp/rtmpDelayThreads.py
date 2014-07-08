import subprocess, time, sys, threading, Queue, datetime
import rtmpDelay

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
        
        sData = rtmpDelay.readHeader(pDumper.stdout)
        self.qLock.acquire()
        self.qData.put(sData)
        self.qLock.release()

        while not exitFlag:
            sData = rtmpDelay.readPackage(pDumper.stdout)
            self.qLock.acquire()
            self.qData.put(sData)
            self.qLock.release()
        pDumper.kill()
        
def run(delay):
    try:
        global exitFlag
        print 'Opening stream...'
        sys.stdout.flush()
        
        dataQueue = Queue.Queue()
        queueLock = threading.Lock()
        dumperThread = DumperThread(0, 'dumper', dataQueue, queueLock)
        dumperThread.start()

        while dataQueue.empty():
            time.sleep(0)
            
        print 'Sleeping for %f secs... ' % delay,
        sys.stdout.flush()
        
        #time.sleep(delay)
        iStart = datetime.datetime.now()
        while (datetime.datetime.now() - iStart).total_seconds() < delay:
            time.sleep(0)
            
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
        print "Unexpected error:", sys.exc_info()
        sys.stdout.flush()
        exitFlag = 1
        dumperThread.join()
        pPlayback.kill()
        
if __name__ == '__main__':
    run(float(sys.argv[1]))
