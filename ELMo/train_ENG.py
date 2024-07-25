import tensorflow as tf
#tf.disable_v2_behavior()
import tensorflow_hub as hub
#import tensorflow as tf
from keras import backend as K
import urllib.request
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Dense, Lambda, Input



# 0. Download (Command) and Load Datasets
# urllib.request.urlretrieve("https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv", filename="spam.csv")

data = pd.read_csv('spam.csv', encoding='latin-1')


# 1. Load Model 'ELMo'
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

sess = tf.Session()
K.set_session(sess)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())


# 2. Pre-processing
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])
y_data = list(data['v1'])
X_data = list(data['v2'])

n_of_train = int(len(X_data) * 0.8)
n_of_test = int(len(X_data) - n_of_train)

X_train = np.asarray(X_data[:n_of_train]) #X_data 데이터 중에서 앞의 4457개의 데이터만 저장
y_train = np.asarray(y_data[:n_of_train]) #y_data 데이터 중에서 앞의 4457개의 데이터만 저장
X_test = np.asarray(X_data[n_of_train:]) #X_data 데이터 중에서 뒤의 1115개의 데이터만 저장
y_test = np.asarray(y_data[n_of_train:]) #y_data 데이터 중에서 뒤의 1115개의 데이터만 저장


# 3. Design Model
# 데이터의 이동이 케라스 → 텐서플로우 → 케라스가 되도록 하는 함수
def ELMoEmbedding(x):
    return elmo(tf.squeeze(tf.cast(x, tf.string), axis=1), as_dict=True, signature="default") ["default"]

input_text = Input(shape=(1,), dtype=tf.string)
embedding_layer = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
hidden_layer = Dense(256, activation='relu')(embedding_layer)
output_layer = Dense(1, activation='sigmoid')(hidden_layer)

model = Model(inputs=[input_text], outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# 4. Train Model
# hyper-parameter 
epochs = 10
batchSize = 60

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize)


# 4. Test Model
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))