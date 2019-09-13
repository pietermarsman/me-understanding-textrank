import re

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt

from wikipedia import query_wikipedia
from sentence_preprocessing import split_sentences, filter_sentences, clean_sentence
from pretty_print import print_with_emphasis, print_summary


with open('stopwords.json', 'r') as fin:
    STOP_WORDS = json.load(fin)
    

def sentence_similarity(sentences, stop_word_language=None):
    """Compute similarity between sentences
    
    Using:
    - cosine similarity on
    - character n-grams
    """
    stop_words = STOP_WORDS.get(stop_word_language, None)
    vectorizer = TfidfVectorizer(analyzer='char_wb', min_df=0.05, max_df=0.5, ngram_range=(1, 10), stop_words=stop_words)
    vectors = vectorizer.fit_transform(sentences)
    similarity = metrics.pairwise.cosine_similarity(vectors, vectors)
    return similarity


def edge_weights(similarity, d=0.85):
    """Transform sentence similarities to proper normalized weights
    
    :param d: damping factor, percentage of jumps to random vertex
    """
    weights = similarity
    np.fill_diagonal(weights, 0)
    weights = normalize(weights, 'l1')
    weights = (1 - d) / weights.shape[0] + d * weights
    return weights


def compute_rank_naive(weights, steps=100):
    """Compute rank by iterative updating
    
    Similar to first eigenvector if the number of steps is large enough
    """
    rank = np.ones(weights.shape[0])
    for _ in range(steps):
        rank = np.array([weights[:,i].dot(vertices) for i in range(rank.shape[0])])
    return rank


def compute_rank_eigenvector(weights):
    """Compute rank by eigenvector decomposition"""
    eigenvalue, eigenvector = np.linalg.eig(weights.T)
    ind = eigenvalue.argmax()
    largest_vector = np.abs(eigenvector[:, ind])
    return (largest_vector - largest_vector.min()) / (largest_vector.max() - largest_vector.min())


def summarize_wiki(page, language='nl', n_input_sentences=None, sort=False,
                   min_rank=None, max_rank=None, n_rank_sentences=None):
    """Summarize wikipedia page using TextRank
    
    :param language: wikipedia language to use
    :param n_input_sentences: if not None, uses only the n largest sentences from the wikipedia page
    :param sort: if the output should be sorted on rank
    :param min_rank: filter sentences with minimum rank
    :param max_rank: filter sentences with maximum rank
    :param n_rank_sentences: if not None, show only the n highest ranked sentences
    """
    text = query_wikipedia(page, language)
    sentences = filter_sentences(split_sentences(text), n_input_sentences)
    clean_sentences = list(map(clean_sentence, sentences))

    similarities = sentence_similarity(clean_sentences, stop_word_language=language)
    weights = edge_weights(similarities, d=0.85)
    ranks = compute_rank_eigenvector(weights)
    
    print_summary(sentences, ranks, sort, min_rank, max_rank, n_rank_sentences)