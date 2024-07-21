import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_hub as hub
#import tensorflow as tf
from keras import backend as K
import urllib.request
import pandas as pd
import numpy as np



# 0. Download (Command) and Load Datasets
# urllib.request.urlretrieve("https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv", filename="spam.csv")

data = pd.read_csv('spam.csv', encoding='latin-1')


# 1. Load Model 'ELMo'
elmo = hub.load("https://tfhub.dev/google/elmo/1") #, trainable=True)

sess = tf.Session()
K.set_session(sess)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())


# 2. Pre-processing
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])
y_data = list(data['v1'])
X_data = list(data['v2'])

[0, 0, 1, 0, 0]