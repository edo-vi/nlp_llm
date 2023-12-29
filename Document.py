from collections import Counter, defaultdict
import numpy as np
import nltk
from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
import re
from math import sqrt


# I.e. Document in a Bag Of Word representation
class Document:
    def __init__(self, lines = None, path = None, stemmed = False) -> None:
        if lines is None and path is None:
            raise()
        self._stemmed = stemmed
        self._path = path
        self._counts = Counter()
        self.stemmer = EnglishStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        if path is not None:
            with open(self._path, "r") as f:
                self._lines = f.readlines()
        else:
            self._lines = lines

    @classmethod
    def from_lines(cls, lines, stemmed=False):
        instance = cls("NOPE")
        instance._lines = lines
    
    def tokenize(self, text):
        return nltk.word_tokenize(text)

    def make_bow(self):
        tokens = []
        lines = self.lines()
        for l in lines:
               tok = self.tokenize(l)
               tokens = tokens + tok

        # print(tokens)
        self._counts = Counter(tokens)
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
    
    def cosine_distance(self, other):
        
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
        # Cosine, rounded
        return round(1-(u_dot_v / (norm_u * norm_v)), 4)

    def find_split(self):
        n = int(len(self._lines) / 4)
        d1 = Document(lines=self._lines[0:n]).make_bow()
        d2 = Document(lines=self._lines[n:]).make_bow()
        return d1, d2, d1.cosine_distance(d2)

    def text(self, escape = False):
        t = " ".join(self.lines())
        if escape:
            return re.escape(t)
        else:
            return t
    
    def counts(self):
        return self._counts
    
    def keys(self):
        return list(self._counts.keys())

    def __str__(self):
        return self._counts.__str__()
