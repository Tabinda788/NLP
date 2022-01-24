import spacy
import en_core_web_sm
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.vocab import Vocab
from spacy.language import Language
from spacy import displacy
from spacy.compat import is_config
nlp_lang = English()


nlp_example = en_core_web_sm.load()
my_doc = nlp_example("This is my first example.")
print(my_doc)



#span

span = my_doc[1:6]
print(span)





# token
token = my_doc[4]
print(token)


#tokenizer
blank_tokenizer = Tokenizer(nlp_lang.vocab)
print(blank_tokenizer)


# language
nlp_lang = Language(Vocab())
nlp_lang = English()
print(nlp_lang)


print("------------------------------spacy.explain()--------------------------------")
nlp= en_core_web_sm.load()
spacy.explain("NORP")
doc = nlp("Hello Tabinda")
for word in doc:
   print(word.text, word.tag_, spacy.explain(word.tag_))


print("------------------------------compat--------------------------------")

compat_unicode = unicode_("This is Trialx")
print(compat_unicode)

print("------------------------------displacy--------------------------------")
text = """When Sebastian Thrun started working on self-driving cars at Google in
2007, few people outside of the company took him seriously. But Google is 
starting from behind. The company made a late push into hardware, and Apple's
Siri has clear leads in consumer adoption."""

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
colors = {"ORG": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
options = {"ents": ["ORG"], "colors": colors}
displacy.serve(doc, style="ent", options=options)