from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Flatten


from numpy import argmax
from numpy import array

from nltk.translate.bleu_score import corpus_bleu


from KerasWordEmbedding.PreprocessingUtils import *


#get class objects
obj_prepUtils = PreprocessingUtils()








# define documents
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice Work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']
# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])




vocab_size = obj_prepUtils.get_vocab_size(docs)
hotEncodedDocs = obj_prepUtils.hot_ecode_docs(docs, vocab_size, True)
padAndEncodedDocs = obj_prepUtils.pad_sequences(hotEncodedDocs)


print(obj_prepUtils.get_vocab_size(docs))
print(obj_prepUtils.get_maxLen(padAndEncodedDocs))
print(hotEncodedDocs)
print(padAndEncodedDocs)




model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=obj_prepUtils.get_maxLen(padAndEncodedDocs)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))




# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])



# summarize the model
print(model.summary())



# fit the model
model.fit(padAndEncodedDocs, labels, epochs=50, verbose=0)


# evaluate the model
loss, accuracy = model.evaluate(padAndEncodedDocs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))




