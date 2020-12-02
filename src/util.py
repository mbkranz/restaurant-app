
#sqlalchemy and pandas for data 
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
import pandas as pd
#spacy for tokenization
from spacy.lang.en import English # Create the nlp object
import spacy
#gensim for similarity
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.similarities.docsim import MatrixSimilarity,Similarity
#itertools for getting similarity edges
#networkx for organizing similarities
#plotly for visualization

nlp = spacy.load('en_core_web_sm')

def tokenize_text(text_str,nlp_obj=nlp):
    '''
    use spacy to separate text into words
    (ie tokenization)
    and return the lemmatization 
    (ie feet for footing and foot)
    for only nouns and adjectives
    
    TODO: refine methodology
    '''
    spacy_doc = nlp_obj(text_str)
    
    tokenized_doc = [
        (token.lemma_,token.pos_)
        for token in spacy_doc
        if token.pos_ in ("NOUN","ADJ","PROPN")
        ]
    
    return tokenized_doc
    #return spacy_doc


