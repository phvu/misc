import segmentedLsq
from scipy import misc
import sys
from matplotlib import cm as cm, pyplot as plt, widgets as widgets

SZ = (32, 300)

class SegmentVisualizer(object):
    def __init__(self, data, C):
    
        print 'Initalizing (computing distances), please wait...'
        self.sl = segmentedLsq.SegmentedLsq(data)
        self.data = data
        
        print 'Here we go!'
        plt.ion()
        self.figSegment = plt.figure()
        
        self.currentMeasure = self.sl.distanceMode
        self.axMeasure = self.figSegment.add_subplot(10, 2, 17, axisbg='lightgoldenrodyellow', adjustable='datalim')
        self.axMeasure.set_position([0.1, 0.05, 0.35, 0.2])
        self.radioMeasure = widgets.RadioButtons(self.axMeasure, \
                     labels = ['Minimum Least Squares', 'Average Euclidean'], \
                     active = self.currentMeasure)
                     
        self.axC = self.figSegment.add_subplot(10, 2, 18, axisbg='lightgoldenrodyellow', adjustable='datalim')
        self.axC.set_position([0.51, 0.1, 0.4, 0.05])
        self.sliderC = widgets.Slider(self.axC, label='C', valmin=-0.5, valmax=1, valinit=C)

        self.axSpec = self.figSegment.add_subplot(1, 1, 1)
        #self.axSpec.set_position()
        
        def radio_onClicked(msg):
           self.currentMeasure = [m.get_text() for m in self.radioMeasure.labels].index(msg)
           self.updateSegments()
         
        def slider_onChanged(newVal):
            self.updateSegments()
         
        self.sliderC.on_changed(slider_onChanged)
        self.radioMeasure.on_clicked(radio_onClicked)
        
        self.updateSegments()
        plt.show()
        print 'Press Enter to exit.'
        raw_input()
        
    def updateSegments(self):
        C = self.sliderC.val
      
        if (self.currentMeasure != self.sl.distanceMode):
            print 'Recomputing distance, please wait...'
            self.sl.distanceMode = self.currentMeasure
            self.sl.computeDistance()
            print 'Recomputing distance done.'
            
        bounds = self.sl.Segment(C)[1]
        self.axSpec.cla()
        self.axSpec.imshow(self.data, cmap = cm.Greys_r)
        self.axSpec.vlines(bounds, 0, SZ[0], colors='r')
        #plt.imshow(tag, cmap=pylab.get_cmap('PuBu'), origin="lower", \
        #  interpolation='none', aspect='auto', extent=[0, len(obs)-1, 15, 22])
        #  plt.plot(range(0, len(obs)), obs, lw=3, label='Observation', color='red')
        self.figSegment.canvas.draw()


################################################################################

def readImage(sPath):
    m = misc.imresize(misc.imread(sPath), SZ)
    m = (m - m.min())/float(m.max() - m.min())
    return m
    
def displayGUI(specFile):
    SegmentVisualizer(readImage(specFile), 0)
    
def computeCLI(specFile, C):
    m = readImage(specFile)
    print 'Computing the distances, please wait...'
    sl = segmentedLsq.SegmentedLsq(m)
    
    bounds = sl.Segment(C)[1]
    print 'Got', len(bounds) - 1, 'segments:'
    print bounds
    
    
if __name__ == '__main__':
    if len(sys.argv) == 2:
        displayGUI(sys.argv[1])
    elif len(sys.argv) == 3:
        computeCLI(sys.argv[1], float(sys.argv[2]))
    else:
        print 'Invalid parameters'
