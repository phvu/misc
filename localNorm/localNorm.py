import numpy as np
from scipy import misc
import math
from matplotlib import cm as cm, pyplot as plt, widgets as widgets
import sys

'''
The whole mess of organizing subplots in this script
can be easily overcomed if we use AxesGrid and/or gridspec.
However that requires matplotlib > v1.0,
but by default, I only have matplotlib 0.99.
So the code is a little messy.
'''

def readImage(imgFile):
   img = misc.imread(imgFile)
   return (img / 255.)

def scanAcrossMaps(img, n, power = 1, computeMean = False, means = None):
   sizes = img.shape
   output = np.zeros(sizes)
   for m in xrange(0, sizes[2]):
      startMap = max(0, m - (n/2))
      endMap = min(sizes[2], m + (n/2) + 1)

      tmp = img[:, :, startMap:endMap].copy()
      if means is not None:
         tmp -= np.tile(np.reshape(means[:, :, m], [means.shape[0], means.shape[1], 1]) , [1, 1, endMap - startMap])
      if power != 1:
         tmp = tmp**power
      output[:, :, m] = np.sum(tmp, axis=2)

      if computeMean:
         output[:, :, m] /= (endMap - startMap)
   return output

def scanSameMap(img, n, power = 1, computeMean = False, means = None):
   sizes = img.shape
   output = np.zeros(sizes)
   getStart = lambda x: max(0, x - (n/2))
   getEnd = lambda x, s: min(s, x + (n/2) + 1)

   for y in xrange(0, sizes[0]):
      startY = getStart(y)
      endY = getEnd(y, sizes[0])
      for x in xrange(0, sizes[1]):
         startX = getStart(x)
         endX = getEnd(x, sizes[1])
         tmp = img[startY:endY, startX:endX, :].copy()
         if means is not None:
            tmp -= np.tile(means[y, x, :], [endY - startY, endX - startX, 1])
         if power != 1:
            tmp = tmp**power
         output[y, x, :] = np.sum(np.sum(tmp, axis=0), axis=0)

         if computeMean:
            output[y, x, :] /= (endY - startY)*(endX - startX)
   return output

class ImageNormalizer(object):

   def __init__(self, img):
      self.img = img
      self.n = None
      self.normType = None
      self.denom = None
      self.output = None

   def responseNorm(self, n, k, alpha, beta, scanFunc):
      if self.denom is None:
         self.denom = scanFunc(np.square(self.img), n)
      self.output = self.img / np.power(k + alpha * self.denom, beta)

   def contrastNorm(self, n, k, alpha, beta, scanFunc):
      if self.denom is None:
         means = scanFunc(self.img, n, 1, True)
         self.denom = scanFunc(self.img, n, 2, False, means)
      self.output = self.img / np.power(k + alpha * self.denom, beta)

   def localNorm(self, normType, n, k, alpha, beta):
      if normType == 0:
         scanFunc = scanAcrossMaps
         normFunc = self.responseNorm
      elif normType == 1:
         scanFunc = scanSameMap
         normFunc = self.responseNorm
      elif normType == 2:
         scanFunc = scanAcrossMaps
         normFunc = self.contrastNorm
      elif normType == 3:
         scanFunc = scanSameMap
         normFunc = self.contrastNorm
      else:
         raise ValueError('clgt?')
      
      if normType != self.normType or n != self.n:
         self.normType = normType
         self.n = n
         self.denom = None
      normFunc(n, k, alpha, beta, scanFunc)
      return self.output

class NormDisplayAxes(object):
   def __init__(self, fig, imgFile, counts, location, size, normType, n, k, logAlpha, beta):
      #print imgFile, " at ", "[%d, %d]" % (location[0], location[1]), ", size = [%.3f, %.3f]" % (size[0], size[1])
      self.imgSrc = readImage(imgFile)
      
      # make 3 dimensional image. Feed it to the normalizer
      img = np.empty_like(self.imgSrc)
      img[:] = self.imgSrc
      if len(img.shape) < 3:
         img = np.reshape(img, [img.shape[0], img.shape[1], 1])
      self.imgNormalizer = ImageNormalizer(img)

      margins = [0.05, 0.25]
      self.axImageOrigin = fig.add_subplot(counts[1], counts[0]*3, (counts[1] - location[1] - 1)*counts[0]*3 + location[0]*3 + 1)
      self.axImageNormed = fig.add_subplot(counts[1], counts[0]*3, (counts[1] - location[1] - 1)*counts[0]*3 + location[0]*3 + 2)
      self.axColorBar = fig.add_subplot(counts[1], counts[0]*3, (counts[1] - location[1] - 1)*counts[0]*3 + location[0]*3 + 3)

      self.axImageOrigin.set_position([margins[0] + size[0]*location[0], margins[1] + size[1]*location[1], 0.4*size[0], size[1]])
      self.axImageNormed.set_position([margins[0] + size[0]*location[0] + 0.41*size[0], margins[1] + size[1]*location[1], 0.4*size[0], size[1]])
      self.axColorBar.set_position([margins[0] + size[0]*location[0] + 0.82*size[0], margins[1] + size[1]*location[1] + 0.2*size[1], 0.02*size[0], 0.6*size[1]])
      #self.axImageOrigin.set_xticks([]) 
      #self.axImageOrigin.set_yticks([]) 
      self.axImageNormed.set_xticks([]) 
      self.axImageNormed.set_yticks([]) 
      self.colorBarNormed = None
      self.axImageOrigin.imshow(self.imgSrc, cmap = cm.Greys_r)

   def updateNormedImage(self, normType, n, k, logAlpha, beta):
      normedImg = self.imgNormalizer.localNorm(normType, n, k, 10**logAlpha, beta)
      if normedImg.shape[2] == 1:
         normedImg = np.reshape(normedImg, normedImg.shape[0:2])

      axesImg = self.axImageNormed.imshow(normedImg, cmap = cm.Greys_r)
      self.axColorBar.cla()
      self.colorBarNormed = plt.colorbar(axesImg, cax=self.axColorBar)

class NomalizationVisualizer:

   def __init__(self, normType, n, k, logAlpha, beta, imgFiles):
      
      plt.ion()
      self.figImage = plt.figure()

      # add files
      self.plots = []
      numFiles = len(imgFiles)
      cntRows = int(round(math.sqrt(0.75*numFiles), 0))
      cntCols = int(math.ceil(float(numFiles) / cntRows))
      for i in xrange(0, numFiles):
         self.plots.append(NormDisplayAxes(self.figImage, imgFiles[i], [cntCols, cntRows], \
            [i % cntCols, cntRows - (i / cntCols + 1)], [0.95/cntCols, 0.7/cntRows], normType, n, k, logAlpha, beta))
      

      #self.figImage.subplots_adjust(bottom = 0.3)
      
      self.axNormType = self.figImage.add_subplot(9, 2, 13, axisbg='lightgoldenrodyellow', adjustable='datalim')
      self.axNormType.set_position([0.1, 0.05, 0.4, 0.2])
      self.axN = self.figImage.add_subplot(9, 2, 12, axisbg='lightgoldenrodyellow', adjustable='datalim')
      self.axN.set_position([0.6, 0.2, 0.3, 0.03])
      self.axK = self.figImage.add_subplot(9, 2, 14, axisbg='lightgoldenrodyellow', adjustable='datalim')
      self.axK.set_position([0.6, 0.15, 0.3, 0.03])
      self.axAlpha = self.figImage.add_subplot(9, 2, 16, axisbg='lightgoldenrodyellow', adjustable='datalim')
      self.axAlpha.set_position([0.6, 0.1, 0.3, 0.03])
      self.axBeta = self.figImage.add_subplot(9, 2, 18, axisbg='lightgoldenrodyellow', adjustable='datalim')
      self.axBeta.set_position([0.6, 0.05, 0.3, 0.03])

      self.currentNormType = normType
      self.sliderN = widgets.Slider(self.axN, label='N', valmin=1, valmax=10, valinit=n, valfmt='%d')
      self.sliderK = widgets.Slider(self.axK, label='k', valmin=.1, valmax=10, valinit=k)
      self.sliderAlpha = widgets.Slider(self.axAlpha, label=r'$\log_{10}\alpha$', valmin=-6, valmax=1, valinit=logAlpha)
      self.sliderBeta  = widgets.Slider(self.axBeta, label=r'$\beta$', valmin=0.1, valmax=3, valinit=beta)
      self.radioNormType = widgets.RadioButtons(self.axNormType, \
                     labels = ['ResponseNorm - Across maps', 'ResponseNorm - Same map', \
                               'ContrastNorm - Across maps', 'ContrastNorm - Same map'], active = self.currentNormType)
      self.figTitle = self.figImage.suptitle("Normalization", fontsize=14)

      def radio_onClicked(msg):
         self.currentNormType = [m.get_text() for m in self.radioNormType.labels].index(msg)
         self.updateNormedImage()

      def slider_onChanged(newVal):
         self.updateNormedImage()

      self.sliderN.on_changed(slider_onChanged)
      self.sliderK.on_changed(slider_onChanged)
      self.sliderAlpha.on_changed(slider_onChanged)
      self.sliderBeta.on_changed(slider_onChanged)
      self.radioNormType.on_clicked(radio_onClicked)
      
      self.updateNormedImage()
      plt.show()
      raw_input()

   def updateNormedImage(self):
      (n, k, logAlpha, beta) = (int(self.sliderN.val), self.sliderK.val, self.sliderAlpha.val, self.sliderBeta.val)
      for p in self.plots:
         p.updateNormedImage(self.currentNormType, n, k, logAlpha, beta)

      sTitle = ('%s\n' % self.radioNormType.labels[self.currentNormType].get_text()) + \
               (r'$N=%d,\;k=%0.2f,\;\log_{10}\alpha=%0.2f,\;\beta=%0.2f$' % (n, k, logAlpha, beta))
      self.figTitle.set_text(sTitle)
      self.figImage.canvas.draw()

def displayNorm(normType, n, k, alpha, beta, imgFiles):
   visualizer = NomalizationVisualizer(normType, n, k, alpha, beta, imgFiles)
   
if __name__ == "__main__":
   normType = 1
   (n, k, logAlpha, beta) = (5, 1, -4, 1)
   displayNorm(normType, n, k, logAlpha, beta, sys.argv[1:])
