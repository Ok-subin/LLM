import pandas as pd
import numpy as np
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Reshape, Activation, Input
from tensorflow.keras.layers import Dot
from tensorflow.keras.utils import plot_model
from IPython.display import SVG
import gensim
import tensorflow as tf
from gensim.models.word2vec import Word2Vec

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 1. Load Datasets
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data


# 2. Pre-processing
news_df = pd.DataFrame({'document':documents})

news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z]", " ")
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())

news_df.replace("", float("NaN"), inplace=True)

if (news_df.isnull().values.any()==True):
    news_df = news_df.dropna(how = 'any')

stop_words = stopwords.words('english')
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
tokenized_doc = tokenized_doc.to_list()

drop_train = list([index for index, sentence in enumerate(tokenized_doc) if len(sentence) <= 1])

tokenized_doc = np.delete(tokenized_doc, drop_train, axis=0)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokenized_doc)

word2idx = tokenizer.word_index
idx2word = {value : key for key, value in word2idx.items()}
encoded = tokenizer.texts_to_sequences(tokenized_doc)

vocab_size = len(word2idx) + 1 

# 3. Modify Dataset for Negative Sampling
skip_grams = [skipgrams(sample, vocabulary_size=vocab_size, window_size=10) for sample in encoded]


# 4. Create SGNS (Skip-Gram with Negative Sampling)
# hyper-parameters
embedding_dim = 100
epochs = 50

# embedding tables
w_inputs = Input(shape=(1, ), dtype='int32')
word_embedding = Embedding(vocab_size, embedding_dim)(w_inputs)

c_inputs = Input(shape=(1, ), dtype='int32')
context_embedding  = Embedding(vocab_size, embedding_dim)(c_inputs)

dot_product = Dot(axes=2)([word_embedding, context_embedding])
dot_product = Reshape((1,), input_shape=(1, 1))(dot_product)
output = Activation('sigmoid')(dot_product)


with tf.device('/gpu:0'):
        
    model = Model(inputs=[w_inputs, c_inputs], outputs=output)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam')
    plot_model(model, to_file='model3.png', show_shapes=True, show_layer_names=True, rankdir='TB')

    for epoch in range(1, epochs+1):
        loss = 0
        for _, elem in enumerate(skip_grams):
            first_elem = np.array(list(zip(*elem[0]))[0], dtype='int32')
            second_elem = np.array(list(zip(*elem[0]))[1], dtype='int32')
            labels = np.array(elem[1], dtype='int32')
            X = [first_elem, second_elem]
            Y = labels
            loss += model.train_on_batch(X,Y)  
        print('Epoch :',epoch, 'Loss :',loss)


    # 5. Savd Embedding Vector
    f = open('vectors.txt' ,'w')

    vectors = model.get_weights()[0]
    for word, i in tokenizer.word_index.items():
        f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
    f.close()