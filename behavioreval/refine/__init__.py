"""Refine mode: improve one skill's text automatically, the way the model team improves a model.

rubric = the spec, simulated users = rollouts, judge = reward, pairwise vs the best version = preferences,
editor = one small optimizer step, held-out cards = the eval set, the morning report = human eval.
Entry point: ``python3 -m behavioreval refine <check|new|run|resume|report> <skill>``."""
