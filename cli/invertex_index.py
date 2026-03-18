from movies_path import file, cache_docmap, cache_index
import pickle
import json, os
import string #, json
from movies_path import *
from nltk.stem import PorterStemmer

class InvertedIndex:
    def __init__(self) -> None:
        self.index = {} # {str:set(list[str])}
        self.docmap = {} # {int:[str]}
    
    def __add_document(self, doc_id: int, text: str):
        token_text = tokenization(text_processor(text))
        for t in token_text:
            if t not in self.index:
                self.index[t] = set()
            self.index[t].add(doc_id) #No hay que asignar la operación self.index[t].add() a nada porque esta da como resultado None
                                      #Es decir, si haces self.index[t] =self.index[t].add(), self.index[t] va a ser None
                                      #Porque estás reasignando el valor en el diccionario

    def get_documents(self, term) -> list[int]:
        lower_version = term.lower()
        if lower_version not in self.index:
            return []
        return sorted(self.index[lower_version])

    
    def build(self):
        with open(file) as f:
            movies = json.load(f)
        for m in movies["movies"]:
            doc = f"{m['title']} {m['description']}"
            self.__add_document(m["id"], doc)
            self.docmap[m["id"]] = m
    
    def save(self):
        if not os.path.isdir(os.path.dirname(cache_docmap)):
           try:
               os.mkdir(os.path.dirname(cache_docmap))
           except Exception as e:
               print("Error: ", e)
               return
        with open(cache_docmap, "wb") as cd:
            pickle.dump(self.docmap, cd)
        with open(cache_index, "wb") as ci:
            pickle.dump(self.index, ci)


    def load(self):
        try:
            with open(cache_index, 'rb') as f:
                self.index = pickle.load(f)
            with open(cache_docmap, 'rb') as fc:
                self.docmap = pickle.load(fc)
        except Exception as e:
            raise Exception("Ocurrió un error: ", e)



def text_processor(s: str) -> str:
    return s.lower().translate(str.maketrans('', '', string.punctuation))

def tokenization(s: str) -> list[str]:
    stemmer = PorterStemmer()
    tokens = list(filter(None, s.split(" ")))
    with open(stop_words_file) as f:
        content = f.read()
    list_sw= content.splitlines()
    new_tokens= []
    for t in tokens:
        if t not in list_sw:
            new_tokens.append(stemmer.stem(t))
    return new_tokens

def search_helper(II: InvertedIndex, query: str, limit: int = DEFAULT_LIMIT) -> list[dict]:
    processed_query = tokenization(text_processor(query))
    print(processed_query)
    ids = []
    results = []
    for t in processed_query:
        ids.extend(II.get_documents(t))
    for i in range(limit):
        results.append(II.docmap[ids[i]])
    return results