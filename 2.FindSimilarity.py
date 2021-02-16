# %%
'''
Ref : https://lukkiddd.com/paragraph-vector-what-is-it-8bcc44d83cfb
'''
import gensim

# %%
'''
Load The Pretrain Doc2Vec Model
'''
model = gensim.models.doc2vec.Doc2Vec.load('./model_docvec.d2v')

# %%

# %%
'''
Test the model !!
By get input document and transform it into infer vector
'''

# doc_org = 'ขอ|พึ่ง|แรง|แห่ง|ฝัน|บันดาล|ให้|เจอ|คนดี|ให้|สาว|ไกล|บ้าน|คน|นี้|ได้|มี|คน|คอย|ปลอบ|เหงา|งาน|ยุ่ง|เมือง|ใหญ่|ขอ|ใจ|ฮัก|มั่น|กัน|หนาว|นำพา|ใน|ทุก|เรื่องราว|ให้|สาว|ดอก|หญ้า|อุ่นใจ'

# ตัดทอน Tokens บางตัวออกไป
# id : 0
# doc_input = 'ขอ|พึ่ง|แรง|แห่ง|ฝัน|ให้|เจอ|คนดี|ให้|สาว|ไกล|บ้าน|นี้|ได้|มี|คน|คอย|ปลอบ|เหงา|งาน|ยุ่ง|เมือง|ใหญ่|ขอ|ใจ|ฮัก|มั่น|กัน|หนาว|นำพา|ทุก|เรื่องราว|ให้|ดอก|หญ้า|อุ่นใจ'

# unknow_id
doc_input = 'ทุกข์|ท้อ|มี|บ้าง|เมื่อ|เข้ามา|ย่าง|อยู่|เมืองหลวง|แม่|อย่า|ได้|คอย|เป็นห่วง|ลูก|มา|เพื่อ|ทวงสิทธิ์|ความจน|สิ|จำ|คำ|เว้า|ให้|ลูก|อด|เอา|เด้อ|หล่า|อย่า|บ่น|ให้|ฮู้|อด|ทน|ย่อ|เด้อ|ลูก|หล่า|แม่'

test_vec = doc_input.split("|")

inferred_vector = model.infer_vector(test_vec)

sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

# Compare and print the most/median/least similar documents from the train corpus
print('Test Document: «{}»\n'.format(' '.join(test_vec)))

print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    # print(u'%s %s: «%s»\n' % (label, sims[index], ''.join(train_corpus[sims[index][0]].words)))
    print(u'%s %s: «%d»\n' % (label, sims[index], sims[index][0]))

# ---------------------------------------------------------------------------------------

# %%