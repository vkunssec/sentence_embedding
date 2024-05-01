from collections import Counter
import math
import string
from typing import List

import spacy

from normalize_map import normalize_map


nlp = spacy.load("pt_core_news_sm", disable=['ner', 'textcat'])


class Utils:
    def __init__(self) -> None:
        self.nlp = nlp


    def filtering(self, text: str) -> List[str]:
        self.nlp.max_length = 15000000
        doc = self.nlp(text)
        
        def _valid(word):
            lexeme = self.nlp.vocab[word]
            return not lexeme.is_stop and lexeme.is_alpha

        return [ word for word in [ token.text for token in doc ] if _valid(word) ]


    def lemmatization(self, arr: List[str]) -> List[str]:
        text = " ".join(arr)
        doc = self.nlp(text)
        return [ token.lemma_ for token in doc ]
    
    
    def normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = text.translate(str.maketrans(normalize_map))
        text = text.translate(text.maketrans("", "", string.punctuation))
        return text


    def TF(self, text: str) -> dict[str, float]:
        text = self.normalize(text)
        tf_dic = Counter(text.split())
        return { word: tf_dic[word] / len(word) for word in tf_dic }


    def IDF(self, tf_dic: List[dict]) -> dict[str, float]:
        counter = {}
        for dic in tf_dic:
            for word, _ in dic.items():
                if word in counter:
                    counter[word] += 1
                    continue
                counter[word] = 1

        return { word: math.log(len(tf_dic) / counter[word]) for word in counter }


    def _TF_IDF(self, tf_dic: dict, idf_dic: dict) -> dict[str, float]:
        return { word: tf_dic[word] * idf_dic[word] for word in tf_dic }


    def TF_IDF(self, corpus: List[str]):
        list_dic = [ self.TF(text) for text in corpus ]
        idf_dic = self.IDF(list_dic)
        return [ self._TF_IDF(tf_dic=tf, idf_dic=idf_dic) for tf in list_dic ]


