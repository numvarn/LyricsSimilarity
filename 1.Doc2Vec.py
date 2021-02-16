# %%
import gensim
import os
import pandas as pd
import random

# %%
'''
Read All Documents 
'''
path = "./data/05.lyrics_processed_final.csv"
data = pd.read_csv(path)

# %%
'''
Display Sample of Documents
'''
data.head()

# %%
'''
Display Documents Info.
'''
data.info()

# %%
'''
Use only some past of document (Hook)
'''
documents = data["field_lyrics_hook_processed"].values

# %%
documents[0:3]

# %%
'''
Convert Documents to Train Corpus by Using Gensim Doc2Vec
'''
train_corpus = []
for i, doc in enumerate(documents):
    train_corpus.append(gensim.models.doc2vec.TaggedDocument(doc.split("|"), [i]))

# %%
train_corpus[0:5]

# %%
'''
Create Doc2Vec Model
'''
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)

# %%
'''
Create Vocabulary
'''
model.build_vocab(train_corpus)

# %%
'''
Train model with train corpus
'''
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

# %%
'''
Find Similarity of earch documents
'''
ranks = []
second_ranks = []

for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    
    rank = [docid for docid, sim in sims].index(doc_id)
    
    ranks.append(rank)
    
    second_ranks.append(sims[1])
# ---------------------------------------------------------------------------------------

# %%
'''
Test the model !!
By find similarity of last document
'''
print('Document ({}): «{}»\n'.format(doc_id, ''.join(train_corpus[doc_id].words)))

print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)

for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ''.join(train_corpus[sims[index][0]].words)))

# ---------------------------------------------------------------------------------------

# %%
'''
Test the model !!
By find similarity of radcom id document
Pick a random document from the corpus and infer a vector from the model
'''
doc_id = random.randint(0, len(train_corpus) - 1)

# Compare and print the second-most-similar document
print('Train Document ({}): «{}»\n'.format(doc_id, ''.join(train_corpus[doc_id].words)))
sim_id = second_ranks[doc_id]

print('Similar Document {}: «{}»\n'.format(sim_id, ''.join(train_corpus[sim_id[0]].words)))

# ---------------------------------------------------------------------------------------

# %%
'''
Test the model !!
By get input document and transform it into infer vector
'''

# doc_org = 'ขอ|พึ่ง|แรง|แห่ง|ฝัน|บันดาล|ให้|เจอ|คนดี|ให้|สาว|ไกล|บ้าน|คน|นี้|ได้|มี|คน|คอย|ปลอบ|เหงา|งาน|ยุ่ง|เมือง|ใหญ่|ขอ|ใจ|ฮัก|มั่น|กัน|หนาว|นำพา|ใน|ทุก|เรื่องราว|ให้|สาว|ดอก|หญ้า|อุ่นใจ'

# ตัดทอน Tokens บางตัวออกไป
doc_input = 'ขอ|พึ่ง|แรง|แห่ง|ฝัน|ให้|เจอ|คนดี|ให้|สาว|ไกล|บ้าน|นี้|ได้|มี|คน|คอย|ปลอบ|เหงา|งาน|ยุ่ง|เมือง|ใหญ่|ขอ|ใจ|ฮัก|มั่น|กัน|หนาว|นำพา|ทุก|เรื่องราว|ให้|ดอก|หญ้า|อุ่นใจ'

test_vec = doc_input.split("|")

inferred_vector = model.infer_vector(test_vec)

sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

# Compare and print the most/median/least similar documents from the train corpus
print('Test Document: «{}»\n'.format(' '.join(test_vec)))

print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ''.join(train_corpus[sims[index][0]].words)))

# ---------------------------------------------------------------------------------------
# %%
'''
Save the Doc2Vec model
'''
model.save('./model_docvec.d2v')
print("Trainning the model is finished")

# %%