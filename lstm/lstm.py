# visualize results of Long Short-Term memory (LSTM) cells
# @author: Vu Pham
# Last edit: June 22, 2013

import json
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def parse(fileName, outputFileName):
    with open(fileName, 'r') as f:
        layerInfo = json.load(f)
        
    w = np.matrix(layerInfo['m_W'])
    wDelay1 = np.matrix(layerInfo['m_Wdelay1'])
    wDelay2 = np.matrix(layerInfo['m_Wdelay2'])
    bias = np.matrix(layerInfo['m_b'])

    nGates = 5
    nFilters = w.shape[1] / nGates
    nInput = w.shape[0]
    INPUT_2D_SIZE = [20, 30]
    
    assert(w.shape[1] == wDelay1.shape[1] and w.shape[1] == wDelay2.shape[1])
    assert(w.shape[1] == bias.shape[1] and bias.shape[0] == 1)
    assert((w.shape[1] % nGates) == 0)
    assert(nFilters == wDelay1.shape[0] and nFilters == wDelay2.shape[0])
    
    sizeInput = INPUT_2D_SIZE[0] * INPUT_2D_SIZE[1]
    data = np.ones([INPUT_2D_SIZE[0], INPUT_2D_SIZE[1], nInput])
    output = np.zeros([INPUT_2D_SIZE[0], INPUT_2D_SIZE[1], nFilters])
    state = np.zeros([INPUT_2D_SIZE[0], INPUT_2D_SIZE[1], nFilters])
    
    '''
    INPUT = [nGates*i for i in xrange(0, nFilters)]
    INPUT_GATE = [nGates*i + 1 for i in xrange(0, nFilters)]
    FORGET1_GATE = [nGates*i + 2 for i in xrange(0, nFilters)]
    FORGET2_GATE = [nGates*i + 3 for i in xrange(0, nFilters)]
    OUTPUT_GATE = [nGates*i + 4 for i in xrange(0, nFilters)]
    NOT_INPUT = [i for i in xrange(0, w.shape[1]) if i not in INPUT]
    '''
    INPUT = range(0, nFilters)
    INPUT_GATE = range(nFilters, 2*nFilters)
    FORGET1_GATE = range(2*nFilters, 3*nFilters)
    FORGET2_GATE = range(3*nFilters, 4*nFilters)
    OUTPUT_GATE = range(4*nFilters, 5*nFilters)
    NOT_INPUT = range(nFilters, 5*nFilters)
    
    for i in xrange(0, INPUT_2D_SIZE[0]):
        d1 = np.zeros(nFilters) if i <= 0 else output[i - 1, j, :]
        s1 = np.zeros(nFilters) if i <= 0 else state[i - 1, j, :]
        for j in xrange(0, INPUT_2D_SIZE[1]):
            d2 = np.zeros(nFilters) if j <= 0 else output[i, j - 1, :]
            s2 = np.zeros(nFilters) if j <= 0 else state[i, j - 1, :]
            
            d = (data[i, j, :] * w) + (d1 * wDelay1) + (d2 * wDelay2) + bias
            d = np.squeeze(np.asarray(d))
            d[INPUT] = np.tanh(d[INPUT])
            d[NOT_INPUT] = 1./(1 + np.exp(d[NOT_INPUT]))
            state[i, j, :] = (d[INPUT] * d[INPUT_GATE]) + (d[FORGET1_GATE] * s1) + (d[FORGET2_GATE] * s2)
            output[i, j, :] = state[i, j, :] * d[OUTPUT_GATE]
    
    np.savez(outputFileName, state=state, output=output)
    for i in xrange(0, nFilters):
        plt.imshow(output[:, :, i], cmap=cm.Greys_r)
        plt.show(block=False)
        raw_input()
    

if __name__ == "__main__":
    parse(sys.argv[1], sys.argv[2])
    
    