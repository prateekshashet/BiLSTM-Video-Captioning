from typing import List, Any, Dict, Optional, Union, Sequence
import numpy as np
import torch
import warnings

# Try to import NLTK
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import single_meteor_score
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    warnings.warn("NLTK not available. BLEU and METEOR scores will not be computed.")

# Try to import pycocoevalcap
try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.spice.spice import Spice
    COCOEVAL_AVAILABLE = True
except ImportError:
    COCOEVAL_AVAILABLE = False
    warnings.warn("pycocoevalcap not available. Some metrics will not be computed.")

# Dummy implementations for when dependencies are missing
if not NLTK_AVAILABLE:
    def sentence_bleu(*args, **kwargs):
        warnings.warn("NLTK not available. Returning dummy BLEU score.")
        return 0.0
    
    def single_meteor_score(*args, **kwargs):
        warnings.warn("NLTK not available. Returning dummy METEOR score.")
        return 0.0

if not COCOEVAL_AVAILABLE:
    class DummyScorer:
        def compute_score(self, *args, **kwargs):
            warnings.warn("pycocoevalcap not available. Returning dummy scores.")
            return 0.0, []
    
    Bleu = DummyScorer
    Cider = DummyScorer
    Rouge = DummyScorer
    Meteor = DummyScorer
    Spice = DummyScorer


def compute_bleu(references: List[List[List[int]]], hypotheses: List[List[int]]) -> float:
    """
    Compute BLEU-4 score between references and hypotheses.
    
    Args:
        references: List of reference captions (each a list of token IDs)
                   Shape: [num_samples, num_references, ref_length]
        hypotheses: List of hypothesis captions (each a list of token IDs)
                   Shape: [num_samples, hyp_length]
                   
    Returns:
        BLEU-4 score (float)
    """
    # Convert token IDs to strings for NLTK
    refs = [[[str(token) for token in ref] for ref in sample_refs] 
            for sample_refs in references]
    hyps = [[str(token) for token in hyp] for hyp in hypotheses]
    
    # Compute BLEU-4 score
    smoothie = SmoothingFunction().method4
    bleu_score = 0.0
    
    for i, hyp in enumerate(hyps):
        bleu_score += sentence_bleu(
            refs[i],
            hyp,
            weights=(0.25, 0.25, 0.25, 0.25),  # BLEU-4 weights
            smoothing_function=smoothie
        )
    
    return bleu_score / len(hyps)


def compute_cider(references: List[List[List[int]]], hypotheses: List[List[int]]) -> float:
    """
    Compute CIDEr score between references and hypotheses.
    
    Args:
        references: List of reference captions (each a list of token IDs)
                   Shape: [num_samples, num_references, ref_length]
        hypotheses: List of hypothesis captions (each a list of token IDs)
                   Shape: [num_samples, hyp_length]
                   
    Returns:
        CIDEr score (float)
    """
    # Convert token IDs to strings for pycocoevalcap
    refs = {}
    hyps = {}
    
    for i in range(len(hypotheses)):
        # Convert token IDs to strings
        refs[str(i)] = [
            ' '.join(str(token) for token in ref) 
            for ref in references[i]
        ]
        hyps[str(i)] = [' '.join(str(token) for token in hypotheses[i])]
    
    # Compute CIDEr score
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(refs, hyps)
    
    return score


def compute_metrics(references: List[List[List[int]]], hypotheses: List[List[int]], 
                   metrics: list = ['bleu', 'cider', 'rouge', 'meteor']) -> dict:
    """
    Compute multiple metrics between references and hypotheses.
    
    Args:
        references: List of reference captions (each a list of token IDs)
                   Shape: [num_samples, num_references, ref_length]
        hypotheses: List of hypothesis captions (each a list of token IDs)
                   Shape: [num_samples, hyp_length]
        metrics: List of metrics to compute. Options: 'bleu', 'cider', 'rouge', 'meteor', 'spice'
        
    Returns:
        Dictionary of metric scores
    """
    # Convert token IDs to strings for pycocoevalcap
    refs = {}
    hyps = {}
    
    for i in range(len(hypotheses)):
        # Convert token IDs to strings
        refs[str(i)] = [' '.join(str(token) for token in ref) for ref in references[i]]
        hyps[str(i)] = [' '.join(str(token) for token in hypotheses[i])]
    
    # Initialize scorers
    scorers = {}
    if 'bleu' in metrics:
        scorers['bleu'] = Bleu(4)  # BLEU-4
    if 'cider' in metrics:
        scorers['cider'] = Cider()
    if 'rouge' in metrics:
        scorers['rouge'] = Rouge()
    if 'meteor' in metrics:
        scorers['meteor'] = Meteor()
    if 'spice' in metrics:
        scorers['spice'] = Spice()
    
    # Compute scores
    scores = {}
    
    for name, scorer in scorers.items():
        if name == 'spice':
            # SPICE requires additional processing
            try:
                scores[name], _ = scorer.compute_score(refs, hyps)
            except:
                scores[name] = 0.0
        else:
            try:
                score, _ = scorer.compute_score(refs, hyps)
                if name == 'bleu':
                    scores[name] = score[3]  # BLEU-4 is at index 3
                else:
                    scores[name] = score
            except:
                scores[name] = 0.0
    
    return scores


def compute_grounding_accuracy(grounding_scores: torch.Tensor, 
                             grounding_targets: torch.Tensor) -> float:
    """
    Compute grounding accuracy between predicted and target grounding scores.
    
    Args:
        grounding_scores: Predicted grounding scores (logits)
                         Shape: [batch_size, max_len, num_objects]
        grounding_targets: Target grounding scores (one-hot or soft)
                          Shape: [batch_size, max_len, num_objects]
                          
    Returns:
        Grounding accuracy (float)
    """
    # Convert logits to probabilities
    probs = torch.softmax(grounding_scores, dim=-1)
    
    # Get predicted object indices
    _, pred_objs = torch.max(probs, dim=-1)  # [batch_size, max_len]
    
    # Get target object indices
    _, target_objs = torch.max(grounding_targets, dim=-1)  # [batch_size, max_len]
    
    # Compute accuracy
    correct = (pred_objs == target_objs).float()
    accuracy = correct.mean().item()
    
    return accuracy


def compute_coverage_loss(attention_weights: torch.Tensor, 
                         coverage_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute coverage loss to encourage attention to cover all parts of the input.
    
    Args:
        attention_weights: Attention weights from decoder
                         Shape: [batch_size, max_len, num_pixels]
        coverage_weights: Previous coverage weights (for temporal coverage)
                         Shape: [batch_size, max_len, num_pixels]
                         
    Returns:
        Coverage loss (scalar tensor)
    """
    # Compute coverage for current step
    if coverage_weights is None:
        coverage = torch.zeros_like(attention_weights)
    else:
        coverage = coverage_weights
    
    # Update coverage with current attention weights
    coverage = coverage + attention_weights
    
    # Compute coverage loss (encourage attention to be different from previous steps)
    coverage_loss = torch.sum(torch.min(attention_weights, coverage), dim=-1)  # [batch_size, max_len]
    coverage_loss = coverage_loss.mean()  # Average over sequence and batch
    
    return coverage_loss
