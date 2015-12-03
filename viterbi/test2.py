# -*- coding: utf-8 -*-
"""
Add second test to viterbi algorithm by phvu found here: 

ttp://phvu.net/2013/12/06/sweet-implementation-of-viterbi-in-python/
https://github.com/phvu/misc/tree/master/viterbi

The test replicates the POS tag results from edition 3 of Jurafsky & Martin.
The toy numerical example can be found in section 9.4.3

Currently the whole draft can be found here: 
http://web.stanford.edu/~jurafsky/slp3/

Note: Terminology: 
Jurafsky & Martin : " state observation likelihood" =  "emission probability"
                    " transition probability" =  "trigram probability."

@author: Christian Geng, christian.c.geng@gmail.com
"""

import viterbi
import pandas
from pandas import DataFrame
import nltk
import numpy as np

infile="JurafskyMartinHmmDecode.xlsx"


Apandas = pandas.read_excel(infile,sheetname="Transitions") 
#print Apandas
rownames = Apandas.index.tolist()
A=np.array(Apandas)

Bpandas = pandas.read_excel(infile,'ObsLikelihood')
#print Bpandas
B=np.array(Bpandas)
statenames = Bpandas.index.tolist()


trans=A[1:,:]
pi=np.expand_dims(np.array(A[0,:]),1)
decoder = viterbi.Decoder(pi,trans, B)

""" do the decoding """
states =  decoder.Decode(np.arange(5))
result = np.array(statenames)[states].tolist()
sentence = Bpandas.columns.tolist()
resultTagged = zip(sentence,result)

correct=' Janet/NNP will/MD back/VB the/DT bill/NN'
correct=[nltk.str2tuple(x) for x in correct.split()]
assert (resultTagged==correct)

print "PASSED"
