# Counts the number of parameters in MDLSTM-RNN (Alex Graves)
# @author: Vu Pham
# Last edit: July 22, 2013

import numpy as np

def countParam(lstm1, lstm2, lstm3, nSymbols):
   LSTM_GATES_COUNT = 5
   IMAGE_TILING_SIZE = [2, 2]
   LSTM_FEATURES_1 = lstm1
   CONV_FEATURES_1 = 6
   CONV_FILTER_1 = [2, 4]
   LSTM_FEATURES_2 = lstm2
   CONV_FEATURES_2 = 20
   CONV_FILTER_2 = [2, 4]
   LSTM_FEATURES_3 = lstm3
   SOFTMAX_SIZE = nSymbols

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
   k = LSTM_FEATURES_3 * SOFTMAX_SIZE * 4
   print k
   n += k
   n += SOFTMAX_SIZE
   return n

if __name__ == "__main__":
    print countParam(2, 10, 50, 82)

