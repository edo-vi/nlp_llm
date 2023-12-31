from collections import Counter, defaultdict, namedtuple
import numpy as np
import nltk
from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
import re
from math import sqrt

SplitReturn = namedtuple("SplitReturn", ["left", "right", "size_left", "size_right", "distance"])

# I.e. Document in a Bag Of Word representation
class Document:
    def __init__(self, lines = None, path = None, stemmed = False) -> None:
        if lines is None and path is None:
            raise()
        self._stemmed = stemmed
        self._path = path
        self._tokens = []
        self._counts = Counter()
        self.stemmer = EnglishStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        if path is not None:
            with open(self._path, "r") as f:
                self._lines = f.readlines()
        else:
            self._lines = lines
       
        lines = self._lines
        for l in lines:
            tok = self.tokenize(l)
            tok_ = []
            # Join two consecutive tokens
            i = 0
            while i < len(tok):
                if i + 1 >= len(tok):
                    tok_.append(tok[i])
                else:
                    if tok[i] == tok[i+1]:
                        j = 1
                        new_token = tok[i]
                        #print("! found two identical consecutive tokens :", tok[i])
                        while i+j < len(tok) and tok[i+j] == tok[i]:
                            new_token += f"_{tok[i+j]}"
                            j += 1
                        tok_.append(new_token)
                        i += j
                    else:
                        tok_.append(tok[i])
                i += 1

            self._tokens = self._tokens + tok_

        # print(tokens)
    
    def tokenize(self, text):
        return nltk.word_tokenize(text)

    def make_bow(self):
        self._counts = Counter(self._tokens)
        self.remove_stopwords()
        self.lemmatize()
        return self

    def c(self, key):
        if self._stemmed:
            key = self.stemmer.stem(key.lower()).lower()
        else:
            key = key.lower()
        try:
            return self._counts[key]
        except:
            return 0
        
    def N(self):
        return len(self._tokens)
    
    def contains(self, word):
        return word in self._counts

    def remove_stopwords(self):
        stopwords = nltk.corpus.stopwords.words("english")
        new_counts = {}
        for word in self._counts:
            # If word is a stopword, or has length less than 2, or is ALL special characters
            if word in list(stopwords) or len(word) < 2 or re.match(r"^[_\W]+$", word):
                continue
            else:
                new_counts[word] = self._counts[word]
        del self._counts
        self._counts = Counter(new_counts)

    def stem(self):
        return self.lemmatize()

    def lemmatize(self):
        self._stemmed = True

        stems = dict()
        for word in self._counts.keys():
            value = self._counts[word]

            stemmed = self.lemmatizer.lemmatize(word)
            # Could already be present in the temporary dictionary: add the previous value to
            # the counts and then (line 104) update stems[stemmed] with the value (if already present it gets overwritten)
            if stemmed in stems.keys():
                value += stems[stemmed]
            stems[stemmed] = value
        # delete previous counts and add the stemmed/lemmatized version
        del self._counts
        self._counts = Counter(stems)

    def lines(self):
        return self._lines
    
    def distance(self, other):
       # Relegate to cosine distance
       return self.cosine_distance(other)
    
    def cosine_distance(self, other):
        if self.N() == 0 or other.N() == 0:
            return 1
        # First vector count. Build a defaultdict whose first argument is 'int' from the `counts' dict.
        # basically, will return the standard count if the key is present, 0 otherwise. 
        u = defaultdict(int, self._counts)
        # Second vector count. These two hold the counts for the first and second document respectively.
        v = defaultdict(int, other.counts())
        
        # Norm of the two vectors, i.e. the square root of the sum of the counts squared
        norm_u = sqrt(sum([i * i for i in u.values()]))
        norm_v = sqrt(sum([j * j for j in v.values()]))

        # The union of all keys, from 'self' and 'other' (two documents)
        keys = set(self._counts.keys()).union(set(other.counts().keys()))
        # Dot product
        u_dot_v = 0
        for k in keys:
            # Remember: it's 0 if it's not present
            u_dot_v += u[k] * v[k]
        # Cosine distance (1-cosine similarity), rounded
        return round(1-(u_dot_v / (norm_u * norm_v)), 4)

    def split(self, n):
        if self.N() == 1:
            d1 = Document(lines=self._tokens).make_bow()
            d2 = Document(lines=[]).make_bow()
        else:
            d1 = Document(lines=self._tokens[:n]).make_bow()
            d2 = Document(lines=self._tokens[n:]).make_bow()
        return SplitReturn(left=d1, right=d2, size_left=d1.size(), size_right=d2.size(), distance=d1.distance(d2))
    
    def split_half(self):
        n = int(self.N()/2)
        return self.split(n)
        

    def text(self, escape = False):
        t = " ".join(self._tokens)
        if escape:
            return re.escape(t)
        else:
            return t
    
    def size(self): 
        return int(sum([len(s.encode('utf-8')) for s in self._tokens]) / 1)#(1024 * 1024)

    def counts(self):
        return self._counts
    
    def keys(self):
        return list(self._counts.keys())

    def __str__(self):
        return self._counts.__str__()
