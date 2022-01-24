import enum
from tracemalloc import stop
import nltk
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
stopwords_en = stopwords.words('english')
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
porter = PorterStemmer()
wnl = WordNetLemmatizer()
#print(set(stopwords.words('english')))


with open('/home/tabinda/Desktop/NLP/pvr.txt') as myfile:
    data=myfile.read().replace('\n','')



data2 = data.replace("/","")
# print(data2)
for i, line in enumerate(data2.split('\n')):
    if i>10:
        break
    print(str(i) + ':\t' + line)



print("-----------------------------------------------------------------------")


print(sent_tokenize(data2))


for sent in sent_tokenize(data2):
    print(word_tokenize(sent))

print("--------------------------------------------------------------------------")

single_tokenized_lowered = list(map(str.lower,word_tokenize(data2)))
print(single_tokenized_lowered)


print("--------------------------------------------------------------------------")
stopwords_en = set(stopwords.words('english'))


print([word for word in single_tokenized_lowered if word not in stopwords_en])

print("---------------------------------punctuation---------------------------")
print("From string.puctuation:", type(punctuation),punctuation)

stopwords_en_withpunct = stopwords_en.union(set(punctuation))

print([word for word in single_tokenized_lowered if word not in stopwords_en_withpunct])


print("----------------------------------------stemmer-------------------------------------------------")
for word in single_tokenized_lowered:
    print(porter.stem(word))


print("------------------------------lemmetizer-----------------------------------")
for word in single_tokenized_lowered:
    print(wnl.lemmatize(word))


print("-----------------------------------pos_tags----------------------------------")
stopwords =  set(stopwords.words('english'))
tokenized = sent_tokenize(data2)
for i in tokenized:
    wordslist = nltk.word_tokenize(i)
    wordslist = [w for w in wordslist if not w in stopwords]
    tagged = nltk.pos_tag(wordslist)

    print(tagged)