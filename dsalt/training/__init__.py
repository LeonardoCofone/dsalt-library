"""
Questo modulo espone il trainer DSALT, includendo la classe DSALTTrainer e la funzione di schedule cosine con warmup.
"""
from dsalt.training.trainer import DSALTTrainer, get_cosine_schedule_with_warmup

__all__ = ["DSALTTrainer", "get_cosine_schedule_with_warmup"]