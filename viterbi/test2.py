# -*- coding: utf-8 -*-
"""
Add further test to viterbi algorithm found yyyyy

The test replicates the results from edition 3 of Jurafsky & Martin XXX
Currently it can be found here: 
http://web.stanford.edu/~jurafsky/slp3/
The toy numerical example can be found in section 9.4.3

Terminology: 
Jurafsky & Martin : " state observation likelihood" =  "emission probability"
                    " transition probability" =  "trigram probability."


@author: Christian Geng, christian.c.geng@gmail.com
"""

import viterbi
import pandas
from pandas import DataFrame
import nltk

infile="JurafskyMartinHmmDecode.xlsx"


Apandas = pandas.read_excel(infile,sheetname="Transitions") 
print Apandas
rownames = Apandas.index.tolist()
A=np.array(Apandas)

Bpandas = pandas.read_excel(infile,'ObsLikelihood')
print Bpandas
B=np.array(Bpandas)
statenames = Bpandas.index.tolist()


trans=A[1:,:]
pi=expand_dims(np.array(A[0,:]),1)
decoder = viterbi.Decoder(pi, trans, B)

""" do the decoding """
states =  decoder.Decode(arange(5))
result = array(statenames)[states].tolist()
sentence = Bpandas.columns.tolist()
resultTagged = zip(sentence,result)

correct=' Janet/NNP will/MD back/VB the/DT bill/NN'
correct=[nltk.str2tuple(x) for x in correct.split()]
assert (resultTagged==correct)

print "PASSED"

#T = A.shape[1] # koennte auch shape[0] Mein Fall: 8,7


#N=B.shape[1] # N State Columns the first for Janet et., in my case 5
             # index state s from  from 1 to N

"""
The Viterbi algorithm sets up a probability matrix, with
one column for each observation t and one row for each state in the state graph.
Each column thus has a cell for each state qi in the single combined automaton for
the four words.
"""
#V = np.zeros((N+2,T))
#V = np.zeros((T,N))


#The correct series of tags is:
#(9.15) Janet/NNP will/MD back/VB the/DT bill/NN





""" initialization step """
#  .28* .000032 = .000009
#A[0,s]*B[0,0]
#A[0,:]*B[:,0]
#intoV = A[0,:]*B[:,0]
#V[:,0]=A[0,:]*B[:,0]
#BP[:,1]=0


""" recursion step """
#for t in range(1,T-2): 
#    V[:,t]=np.max(V[:,t-1]) * B[:,t]
#    print pandas.DataFrame(V)
    #for s in range(0,N): print s

#BP=V.argmax(axis=1).tolist()
#tags=Bpandas.index.tolist()
#solution = [tags[x] for x in BP]

   
""" 
Iteration 1 
v2(2) = max * .308 = .0000002772 
v2(3)= max  *.000028 =  2.5e-11 
v2(5)= max * .0002  = .0000000001
"""
#s=0
#np.max(V[:,0]) * B[:,1] # hier sollte man vielleicht nur die Bs nehmen, die grösser sind als 0
#V[:,1]=np.max(V[:,0]) * B[:,1]
#print pandas.DataFrame(V)
# ok    
#t=1
#V[:,t]=np.max(V[:,0]) * B[:,1]
#V[:,t]=np.max(V[:,t-1]) * B[:,t]
#print pandas.DataFrame(V)





""" 
Iteration 2 
v3(3)= max * .00067
v3(4)= max * .00034
v3(5)= max * .000223
v3(6)= max * .0104
"""
#t=2
#V[:,t]=np.max(V[:,t-1]) * B[:,t]
# V[:,t]=np.max(V[:,t-1]) * B[:,t-1]
#print pandas.DataFrame(V)
    
    #V[:,t]=np.max(V[:,0]) * B[:,1]
    
    
 


#transition probabilities P(ti |ti−1) computed from the WSJ corpus without
# smoothing. Rows are labeled with the conditioning event; thus P(V B|MD) is 0.7968.


#The Viterbi algorithm sets up a probability matrix, with
#one column for each observation t and one row for each state in the state graph.
#Each column thus has a cell for each state qi in the single combined automaton for
#the four words.


# The algorithm first creates N = 5 state columns,
#N=len(obslik.columns)


#decoder = viterbi.Decoder(pi, transitions, obslik)


