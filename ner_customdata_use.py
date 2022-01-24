from tkinter.ttk import Style
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
from spacy import displacy

nlp = spacy.blank("en") # load a new spacy model
db = DocBin() # create a DocBin object



import json
f = open('training_data.json')
TRAIN_DATA = json.load(f)


for text, annot in tqdm(TRAIN_DATA['annotations']): 
    doc = nlp.make_doc(text) 
    ents = []
    for start, end, label in annot["entities"]:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents 
    db.add(doc)

#db.to_disk("./training_data.spacy") # save the docbin object





# run the following commands
# python -m spacy init config config.cfg --lang en --pipeline ner --optimize efficiency
# python -m spacy train config.cfg --output ./ --paths.train ./training_data.spacy --paths.dev ./training_data.spacy



# here validation data is same as training data as we can see in the command above
# also the pipeline is for cpu 

nlp_ner = spacy.load("./model-best")

tokens = nlp_ner("The PVR ltd is awsome Rs 25455")

for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)


doc = nlp_ner('''The PVR ltd is awsome Rs 25455 applied informatics''')
print("--------------------------------------------------------------------------")
print(doc.ents)

def extracted_features(extracted_text):
    
    doc = nlp_ner(extracted_text)
    companyname = 0
    money = 0
    for i in range(0, len(doc.ents)):
        if doc.ents[i].label_ == 'COMPANY':
            companyname = doc.ents[i].text
        elif doc.ents[i].label_ == 'MONEY':
            money = doc.ents[i].text
        else:
            pass
            
    data = {'companyname': companyname, 'money': money}
    return data


extracted_text = '''The PVR ltd is awsome Rs 295959595959 Rs 49599 is bal'''


features = extracted_features(extracted_text)
print(features)


#spacy.displacy.render(doc, style="ent")
#displacy.serve(doc, style="ent")