"""
This code defines two functions: calculate_metrics and get_bert_score. calculate_metrics calculates the evaluation metrics (ROUGE and BERTScore) for candidate summaries against reference summaries, while get_bert_score calculates the BERTScore for candidate summaries against reference summaries. Both functions accept lists of candidate and reference summaries as input and return the respective metric scores.
"""


from rouge import Rouge
from bert_score import score
import torch

def calculate_metrics(ref_sentences, cand_sentence):
    """
    Calculate evaluation metrics (ROUGE and BERTScore) for candidate summaries 
    against reference summaries.

    Args:
        ref_sentences (list): List of reference summaries.
        cand_sentence (list): List of candidate summaries.

    Returns:
        tuple: ROUGE-1 F1 score, ROUGE-2 F1 score, ROUGE-L F1 score, and BERTScore F1 score.
    """
    # Calculate the ROUGE scores
    rouge = Rouge()
    rouge_scores = rouge.get_scores(cand_sentence, ref_sentences, avg=True)

    # Calculate the BERTScore
    bertscore = get_bert_score(cand_sentence, ref_sentences)

    return (
        rouge_scores[0]['rouge-1']['f'],
        rouge_scores[0]['rouge-2']['f'],
        rouge_scores[0]['rouge-l']['f'],
        bertscore
    )

def get_bert_score(cands, refs):
    """
    Calculate the BERTScore for candidate summaries against reference summaries.

    Args:
        cands (list): List of candidate summaries.
        refs (list): List of reference summaries.

    Returns:
        float: BERTScore F1 score.
    """
    assert len(cands) == len(refs)
    
    # Calculate the BERTScore
    P, R, F1 = score(cands, refs, lang='en')
    
    # Compute the mean of the scores
    P = torch.mean(P, dim=0).item()
    R = torch.mean(R, dim=0).item()
    F1 = torch.mean(F1, dim=0).item()

    return F1
    
