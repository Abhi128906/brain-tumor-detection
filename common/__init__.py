"""Shared code used by BOTH training scripts and the backend.

Keeping the classifier architecture and the preprocessing in one place guarantees
that what the model saw during training is exactly what it sees at inference time.
"""
