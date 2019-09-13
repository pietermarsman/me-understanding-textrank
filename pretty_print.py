import numpy as np
from IPython.core.display import display, HTML

def print_with_emphasis(text, emphasis=1.0):
    display(HTML('<p style="opacity: {};">{:.2f} \t {}</p>'.format(emphasis, emphasis, text)))
    
def print_summary(sentences, ranks, sort=False, min_rank=None, max_rank=None, max_sentences=None):
    ranked_sentences = list(zip(ranks, sentences))
    
    if sort:
        ranked_sentences = list(reversed(sorted(ranked_sentences)))
        
    if max_rank is not None:
        ranked_sentences = [(rank, sentence) for rank, sentence in ranked_sentences if rank <= max_rank]
        
    if max_sentences is not None:
        ranks, _ = zip(*ranked_sentences)
        if min_rank is None:
            min_rank = 0.0
        if isinstance(max_sentences, int):
            min_rank = max(min_rank, np.sort(ranks)[-max_sentences])
        else:
            min_rank = max(min_rank, np.quantile(ranks, 1-max_sentences))
            
    if min_rank is not None:
        ranked_sentences = [(rank, sentence) for rank, sentence in ranked_sentences if rank >= min_rank]
    
    for rank, sentence in ranked_sentences:
        print_with_emphasis(sentence, rank)