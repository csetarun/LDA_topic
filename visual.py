import json
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import numpy as np
tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    
# create sample documents
doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health." 

# compile sample documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word = dictionary, passes=20)
#print(ldamodel.print_topics(num_topics=2, num_words=10))
topics_matrix = ldamodel.show_topics(formatted=False, num_words=20)
topics_matrix = np.array(topics_matrix,dtype=object)
topic_words = topics_matrix[:,1]
j=0;
'''for i in topic_words:
    print "topic",j
    print [str(word[0]) for word in i]
    j=j+1'''
data={}
child=[]
for i in topic_words:
    vis_topics={}
    nodes=[]
    for word in i:
        d={}
        d['name']=str(word[0])
        d['size']=int(word[1]*100000)
        nodes.append(d)
    vis_topics['name']='topic'+`j`
    vis_topics['children']=nodes
    child.append(vis_topics)
    j=j+1;
data['name']='Visualization'
data['children']=child
json_data=json.dumps(data,indent=4);
file_open=open('custom.json','w')
file_open.write(json_data)
file_open.close()

for doc_count in range(len(doc_set)): #topic id,prob.,'....topic distribution for each document 
    '{},{}'.format(doc_count,max([i[doc_count][1] for  i in ldamodel.get_document_topics(corpus)]))

for i in range(ldamodel.num_topics):
	ldamodel.get_topic_terms(i) #word id,probability for each topic
