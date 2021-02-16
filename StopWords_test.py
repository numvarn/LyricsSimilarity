import pythainlp
from pythainlp import word_tokenize

th_stop = tuple(pythainlp.corpus.common.thai_stopwords())

print(len(th_stop))