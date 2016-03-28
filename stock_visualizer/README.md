# Visualizing stock data using sequence embedding

## Approach

- Take ~1800 series on NASDAQ North US between 2008 and 2015
- Run some "magic" feature engineering (c.f. @minhtran) to create 28 features for each symbol
- Split all the stock into sequences of length 40 (days)
- Train a [Seq2Seq model](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/translate.py)
 to predict the stock feature vector of the next day, given the input.
- The final hidden state (of the encoder) is used as a "vector representation" of the 40-day segment.
The dimension of this vector is equal to the number of hidden units in the RNN (1024 in our case)
- t-SNE all the hidden states.

## How to use

To train the RNN:

    $ python train.py --learning_rate=0.1
    
After training, the feature vectors can be extracted by:

    $ python train.py --decode
    
# Results

We trained a Seq2Seq model of 1024 GRU units, with learning rate of 0.1. 
The final visualization can be seen [here](http://nbviewer.jupyter.org/github/phvu/misc/blob/master/stock_visualizer/Visualize.ipynb)
 
## Next steps

- Use LSTM instead of GRU.
- Use a more sophisticated Seq2Seq model (i.e. more layers, attention models...)
- Maybe some clever strategy in splitting the series/feature engineering
- Fix bugs (if any)