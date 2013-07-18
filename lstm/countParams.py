# Counts the number of parameters in MDLSTM-RNN (Alex Graves)
# @author: Vu Pham
# Last edit: July 17, 2013

import numpy as np

def countStandard(lstm1, lstm2, lstm3, numOutput):
   # the standard architecture with 6, 20 convolutial features,
   # the filters are [2, 2], [2, 4], [2, 4]
   # and the output has numOutput units
   # lstm1, lstm2, lstm3 are the number of features at 3 MDLSTM layers.
   return ( 40*lstm1*lstm1 + 292*lstm1 + 40*lstm2*lstm2 + 780*lstm2 + \
            40*lstm3*lstm3 + 420*lstm3 + numOutput*lstm3 + numOutput)

def countParam():
   LSTM_GATES_COUNT = 5
   IMAGE_TILING_SIZE = [2, 2]
   LSTM_FEATURES_1 = 2
   CONV_FEATURES_1 = 6
   CONV_FILTER_1 = [2, 4]
   LSTM_FEATURES_2 = 10
   CONV_FEATURES_2 = 20
   CONV_FILTER_2 = [2, 4]
   LSTM_FEATURES_3 = 50
   SOFTMAX_SIZE = 82

   lstmParam = lambda di, dr: (LSTM_GATES_COUNT * (di + 2*dr + 1) + 0) * dr * 4
   convParam = lambda si, so, sf: si * so * np.prod(sf) * 4
   
   n = 0
   k = lstmParam(np.prod(IMAGE_TILING_SIZE), LSTM_FEATURES_1)
   print k
   n += k
   k = convParam(LSTM_FEATURES_1, CONV_FEATURES_1, CONV_FILTER_1)
   print k
   n += k
   k = lstmParam(CONV_FEATURES_1, LSTM_FEATURES_2)
   print k
   n += k
   k = convParam(LSTM_FEATURES_2, CONV_FEATURES_2, CONV_FILTER_2)
   print k
   n += k
   k = lstmParam(CONV_FEATURES_2, LSTM_FEATURES_3)
   print k
   n += k
   k = LSTM_FEATURES_3 * SOFTMAX_SIZE
   print k
   n += k
   n += SOFTMAX_SIZE
   return n

if __name__ == "__main__":
    print countParam()

