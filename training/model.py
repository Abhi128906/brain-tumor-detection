"""
Deep Learning Model Architecture for Brain Tumor Classification & Explainability (Grad-CAM).
The network itself now lives in common/classifier_net.py (shared with the FastAPI backend) so that
training and inference can never drift apart. This module re-exports it and keeps the clinical text.
"""

import os
import sys

# Make the repo root importable so `common` resolves no matter where the script is launched from
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.classifier_net import BrainTumorClassifier, BACKBONES  # noqa: F401  (single shared architecture)

CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
CLASS_DISPLAY = {
    'glioma': 'Glioma',
    'meningioma': 'Meningioma',
    'no_tumor': 'No Tumor (Healthy)',
    'pituitary': 'Pituitary Tumor'
}

TUMOR_INFO = {
    'glioma': {
        'name': 'Glioma',
        'has_tumor': True,
        'severity': 'High (Requires Oncological Staging)',
        'origin': 'Glial cells (astrocytes, oligodendrocytes, ependymal cells)',
        'description': 'Gliomas originate in the supportive glial cells of the brain. They can range from slow-growing low-grade astrocytomas to aggressive high-grade glioblastomas (GBM). They frequently exhibit diffuse margins and infiltrative borders.',
        'clinical_actions': 'Contrast MRI/Spectroscopy, neurosurgical consult, biopsy, potential surgical resection followed by radiation therapy and Temozolomide chemotherapy.'
    },
    'meningioma': {
        'name': 'Meningioma',
        'has_tumor': True,
        'severity': 'Moderate (Typically Benign, Watch for Mass Effect)',
        'origin': 'Meninges (dura mater, arachnoid mater)',
        'description': 'Meningiomas arise from the meningeal protective layers surrounding the brain and spinal cord. Approximately 85-90% are benign (WHO Grade I), well-circumscribed, and slow-growing, though they can cause symptoms through compression of adjacent brain tissue.',
        'clinical_actions': 'High-resolution volumetric MRI follow-up. For symptomatic or expanding lesions, complete microsurgical resection (Simpson grading) or stereotactic radiosurgery (Gamma Knife).'
    },
    'pituitary': {
        'name': 'Pituitary Tumor',
        'has_tumor': True,
        'severity': 'Moderate (Endocrine & Visual Field Evaluation Needed)',
        'origin': 'Pituitary gland (sella turcica at skull base)',
        'description': 'Pituitary adenomas arise in the pituitary gland nestled in the sella turcica. Most are benign but can secrete excess hormones (prolactin, ACTH, GH) or cause bitemporal hemianopsia through mass-effect compression of the optic chiasm.',
        'clinical_actions': 'Complete endocrinology panel (prolactin, IGF-1, cortisol), formal visual field perimetry testing, transsphenoidal endoscopic surgical removal or dopamine agonist therapy (e.g., Cabergoline).'
    },
    'no_tumor': {
        'name': 'No Tumor (Normal Scan)',
        'has_tumor': False,
        'severity': 'Normal / Negative',
        'origin': 'Healthy cerebral parenchyma',
        'description': 'No evidence of intracranial mass, abnormal contrast enhancement, abnormal midline shift, or ventricular distortion detected in this MRI slice. Parenchyma, sulci, and gyri appear normal for age.',
        'clinical_actions': 'Routine follow-up or symptom correlation if clinical neurological deficits persist.'
    }
}
