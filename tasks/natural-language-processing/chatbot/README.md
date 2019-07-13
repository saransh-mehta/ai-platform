# Generative_chatbot_seq2seq_tf

This is the code assessment submitted by Saransh Mehta as a part of the process for Produvia. 

I have build an encoder-decoder sequence2sequence model for modelling a chatbot using the twitter chat dataset. The chat dataset is taken from 
https://github.com/marsan-ma/chat_corpus


Encoder-decoder, seq2seq based generative chatbot using Native Tensorflow
The Sequence to Sequence model (seq2seq) consists of two RNNs - an encoder and a decoder.
The encoder reads the input sequence, word by word and emits a context (a function of final hidden state of encoder),
which would ideally capture the essence (semantic summary) of the input sequence.
Based on this context, the decoder generates the output sequence, one word at a time while looking at the context and the previous word during each timestep.

<img src = "images/seq2seq.png">

# Data Preparation
For the seq2seq chatbot, twitter chat log dataset is used obtained from https://github.com/Marsan-Ma/chat_corpus/ and for general pre-processing of data, data.py is used which is also taken from Marsen-Ma. It contains simple functions to filter out sentences

Processed data contains following things

idx_q.npy
idx_a.npy
metadata.pkl
    w2idx
    idx2w
    limit : { maxq, minq, maxa, mina }
    freq_dist

idx_q -> numpy array containing arrays of questions converted into index using w2idx which is word-index vocab idx_a ->numpy array containing arrays of corresponding answers converted into index using w2idx which is word-index vocab maxq -> max length of sen in questions minq -> min length of sen in questions

maxa -> max length of sen in answers mina -> min length of sen in answers

freq_dist -> frequency dist of the words in vocab.

# Vocab Size
Vocab size is important as seq2seq model is gonna compute softmax over vocab_size, hence lesser the size, faster the network. We need to precisely select a value such that it is not too less (else model will output a lot of 'unk') and not too high

Thus, we will plot the freq of most frequent 10k words and see where is the frquency becoming too less, we can chooose that as cut-off point of vocab in the data.py script

<img src = "images/vocab_frequency.jpg">

# Model
I have made a list of placeholders, i.e one placeholder for each time-step
this list of placeholder is having max time steps (20) number of placeholders 
each placeholder will hold the batch, that's y shape (None,) for that time steps
<br><br>
Now decoder inp will be actually one time step behind the decoder output and just
the first time step of decoder inp will contain <'GO'> token. 'go' is simply represented by 0, which is '_'
NOTE:- decoderInp is not a placeholder, it will be created shifting encoderInp placeholder.
<br>
Tensorflow has two major functions which can be used for implementing Seq2Seq in tensorflow

outputs, states = basic_rnn_seq2seq(encoder_inputs, decoder_inputs, cell)
and
outputs, states = embedding_rnn_seq2seq(
			encoder_inputs, decoder_inputs, cell, num_encoder_symbols, num_decoder_symbols,
			embedding_size, output_projection=None, feed_previous=False)

Second functions handles the embedding part also,
			As decoder uses the out produced at t-1 to feed along with input for time step t,
			during training, since the decoder will produce wrong result at t-1, we have to still use 
			the target value provided through decoder_input_placeholder for training.
			This is done by keeping feed_previous = False in the above function
<br>
When it is true, tf only uses the first element of decoder inp to initialize decoder and
then feeds consecutive outputs.

