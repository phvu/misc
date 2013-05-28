import numpy as np
from matplotlib import cm as cm, pyplot as plt, widgets as widgets
from scipy import misc
import sys

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

class ImageNormalizer:

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

class NomalizationVisualizer:

   def __init__(self, imgFile, normType, n, k, alpha, beta):
      self.imgSrc = readImage(imgFile)
      
      # make 3 dimensional image. Feed it to the normalizer
      img = np.empty_like(self.imgSrc)
      img[:] = self.imgSrc
      if len(img.shape) < 3:
         img = np.reshape(img, [img.shape[0], img.shape[1], 1])
      self.imgNormalizer = ImageNormalizer(img)

      plt.ion()
      self.figImage = plt.figure()
      self.axImageOrigin = self.figImage.add_subplot(121)
      self.axImageNormed = self.figImage.add_subplot(122)
      self.figImage.subplots_adjust(bottom = 0.3)
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
      self.axImageOrigin.set_title('Original image')
      self.sliderN = widgets.Slider(self.axN, label='N', valmin=1, valmax=10, valinit=n, valfmt='%d')
      self.sliderK = widgets.Slider(self.axK, label='k', valmin=.1, valmax=10, valinit=k)
      self.sliderAlpha = widgets.Slider(self.axAlpha, label='alpha', valmin=0, valmax=3, valinit=alpha)
      self.sliderBeta  = widgets.Slider(self.axBeta, label='beta', valmin=0.1, valmax=3, valinit=beta)
      self.radioNormType = widgets.RadioButtons(self.axNormType, \
                     labels = ['ResponseNorm - Across maps', 'ResponseNorm - Same map', \
                               'ContrastNorm - Across maps', 'ContrastNorm - Same map'], active = self.currentNormType)

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
      self.axImageOrigin.imshow(self.imgSrc, cmap = cm.Greys_r)
      plt.show()
      raw_input()

   def updateNormedImage(self):
      (n, k, alpha, beta) = (int(self.sliderN.val), self.sliderK.val, self.sliderAlpha.val, self.sliderBeta.val)
      normedImg = self.imgNormalizer.localNorm(self.currentNormType, n, k, alpha, beta)

      if normedImg.shape[2] == 1:
         normedImg = np.reshape(normedImg, normedImg.shape[0:2])
      self.axImageNormed.imshow(normedImg, cmap = cm.Greys_r)
      sTitle = ('%s\n' % self.radioNormType.labels[self.currentNormType].get_text()) + \
               (r'$N=%d, k=%0.2f, \alpha=%0.2f, \beta=%0.2f$' % (n, k, alpha, beta))
      self.axImageNormed.set_title(sTitle)
      self.figImage.canvas.draw()

def displayNorm(imgFile, normType, n, k, alpha, beta):
   visualizer = NomalizationVisualizer(imgFile, normType, n, k, alpha, beta)
   
if __name__ == "__main__":
   normType = 1
   (n, k, alpha, beta) = (5, 1, 0, 1)
   
   if len(sys.argv) >= 6:
      (n, k, alpha, beta) = (int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
   displayNorm(sys.argv[1], normType, n, k, alpha, beta)
