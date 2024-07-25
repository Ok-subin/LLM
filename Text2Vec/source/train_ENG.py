import re
import urllib.request
import zipfile
from lxml import etree
import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


# 0. Download Data (Command)
'''
urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml", filename="ted_en-20160408.xml")
'''


# 1. Pre-processing
targetXML = open('ted_en-20160408.xml', 'r', encoding='UTF8')
target_text = etree.parse(targetXML)

parse_text = '\n'.join(target_text.xpath('//content/text()'))
content_text = re.sub(r'\([^)]*\)', '', parse_text)
sent_text = sent_tokenize(content_text)

normalized_text = []
for string in sent_text:
     tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
     normalized_text.append(tokens)

result = [word_tokenize(sentence) for sentence in normalized_text]


# 2. Train Model
model = Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=4, sg=0)


# 3.Test Model
model_result = model.wv.most_similar("man")
print(model_result)

"""
출력 예시
[('woman', 0.8355434536933899), ('guy', 0.8176618814468384), ('boy', 0.76740562915802), ('lady', 0.7483574748039246), \ 
 ('girl', 0.7273262143135071), ('soldier', 0.7074109315872192), ('gentleman', 0.6947587728500366), ('kid', 0.6841924786567688),87728500366), \ 
 ('kid', 0.6841924786567688), ('surgeon', 0.6824308037757874), ('poet', 0.6695438623428345)]
"""


# 4. Save Model
model.wv.save_word2vec_format('eng_w2v')
