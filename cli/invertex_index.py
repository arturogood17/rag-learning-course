from search import *
from movies_path import file, cache_docmap, cache_index
import pickle
import json, os

class InvertedIndex:
    def __init__(self) -> None:
        self.index = {} # {str:set(list[str])}
        self.docmap = {} # {int:[str]}
    
    def __add_document(self, doc_id: int, text: str):
        token_text = tokenization(text)
        for t in token_text:
            if t not in self.index:
                self.index[t] = set()
            self.index[t].add(doc_id) #No hay que asignar la operación self.index[t].add() a nada porque esta da como resultado None
                                      #Es decir, si haces self.index[t] =self.index[t].add(), self.index[t] va a ser None
                                      #Porque estás reasignando el valor en el diccionario

    def get_documents(self, term) -> list[str]:
        lower_version = term.lower()
        return list(set((sorted(self.index[lower_version]))))
    
    def build(self):
        with open(file) as f:
            movies = json.load(f)
        for m in movies["movies"]:
            doc = f"{m['title']} {m['description']}"
            self.__add_document(m["id"], doc)
            self.docmap[m["id"]] = m
    
    def get(self):
        print("Index: ", self.index)
        print("Doc: ", self.docmap)

    
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