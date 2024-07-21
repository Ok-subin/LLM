import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
from gensim.models import Word2Vec
from tqdm import tqdm


# 0. Download Data (Command)
'''
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
'''


# 1. Pre-processing
train_data = pd.read_table('ratings.txt')   

if (train_data.isnull().values.any()==True):
    train_data = train_data.dropna(how = 'any')

train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
okt = Okt()
tokenized_data = []
for sentence in tqdm(train_data['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    tokenized_data.append(stopwords_removed_sentence)


# 2. Train Model
model = Word2Vec(sentences = tokenized_data, vector_size = 100, window = 5, min_count = 5, workers = 4, sg = 0)


# 3. Test Model
model_result = model.wv.most_similar("최민식")
print(model_result)


# 4. Save Model
model.wv.save_word2vec_format('kor_w2v')