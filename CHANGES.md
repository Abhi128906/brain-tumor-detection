# CHANGES - accuracy, honesty and correctness fixes

## Bugs found and fixed

| # | Problem in the original code | Fix |
|---|---|---|
| 1 | **Classifier output was fabricated.** With no `best_model.pt` the API returned 93-95% "Glioma/Meningioma/Pituitary" based on the *filename* (`*glioma*`) or the mask's position. | `routes.py`: no checkpoint => explicit `is_available:false` ("Classification unavailable"). Never invents probabilities. |
| 2 | **Backend classifier had a different architecture** from `training/model.py`, and `load_state_dict(strict=False)` would silently load *nothing* while reporting "configured". | One shared network in `common/classifier_net.py`; strict loading; class order validated; health endpoint reports whether the checkpoint truly loaded. |
| 3 | **Train/inference preprocessing mismatch.** Classifier received the segmenter's z-scored tensor (trained with ImageNet norm); resizing used cv2 while training used PIL. | `common/preprocessing.py` is the single source of truth; unit test proves identical tensors to the training transform. |
| 4 | **Test set used for checkpoint selection**, no validation split, no patient-level split, 10 epochs from scratch, no augmentation. | New `training/dataset.py` + `train.py`: patient-aware val split, pretrained backbone (EfficientNet-B0 default), warm-up then fine-tune, label smoothing, class weights, augmentation, early stopping on validation, test evaluated once. |
| 5 | **No calibration.** Softmax confidences were uncalibrated. | Temperature scaling fitted on validation logits, stored in the checkpoint; ECE reported before/after. Abstain below `ABSTAIN_THRESHOLD`. |
| 6 | **FastBrain-UNet was never trained** (random weights served as a real model) and its Dice/IoU/memory figures in `/benchmark` were hardcoded literals. Docs claimed 1.15M params; it has 190,860. | Untrained => `503` with instructions. New `training/train_segmentation.py` (distillation). `/benchmark` returns measured latency/params/memory and Dice/IoU read from `models/segmentation_metrics.json` (null if not evaluated). |
| 7 | **MC-dropout uncertainty was never called** by `/analyze`, `UncertaintyViewer` imported a type that did not exist (dead code), and `estimate_mc_uncertainty` called `model.train()` which corrupts BatchNorm. Variance was also rescaled per image. | Wired into `/analyze` and the dashboard; only Dropout layers are switched on; fixed variance scale (`UNCERTAINTY_VAR_SCALE`); BatchNorm stats verified unchanged by test. |
| 8 | **`no_tumor` silently erased the segmentation mask.** | Masks are never modified by the classifier. Disagreements are flagged (`consistency`, human-review reasons). |
| 9 | Flask demo (`app/server.py`) had the same fake `0.92` confidence/heatmap fallback and could serve a random-init model. | Returns `503` without a valid checkpoint; strict, calibrated loading. |
| 10 | `.env.example` settings were never read. In-memory result store grew unbounded. | `backend/app/config.py` reads them; results store capped (`MAX_STORED_RESULTS`). |
| 11 | Dockerfile did not copy the new shared package. | `COPY common/ ./common/` added. |

## Accuracy features added
* ImageNet-pretrained backbones: `mobilenet_v3 | resnet18 | resnet50 | efficientnet_b0`.
* Test-time augmentation (horizontal flip) for classifier and segmenter.
* Segmentation post-processing (speckle removal, hole filling).
* `evaluate.py`: accuracy with bootstrap CI, macro-F1, AUC, ECE, confusion matrix.

## New / changed files
`common/` (new), `backend/app/config.py` (new), `backend/app/services/{postprocess,metrics_store}.py` (new),
`training/train_segmentation.py` (new), `backend/tests/test_fixes.py` (new),
`training/{dataset,train,evaluate,model}.py`, `backend/app/api/routes.py`, `backend/app/models/*`,
`backend/app/services/uncertainty.py`, `app/server.py`, `colab_train.py`, frontend types/panels,
`Dockerfile`, `requirements.txt`, `.env.example`, `README.md`.

## Things you still need to do (cannot be done without your data / GPU)
1. Train the classifier: `python training/train.py` -> creates `models/best_model.pt`.
2. Run `python training/train_segmentation.py --model nnunet --eval-only` to get real Dice/IoU for your existing weights
   (note: those weights were originally selected on the test split, so that number is optimistic).
3. Optionally retrain the nnU-Net and train FastBrain-UNet.
4. `models/model_metadata.json` currently describes a classifier whose weights are not in the repo; `train.py` overwrites it.
5. The notebooks still contain the old training recipe; prefer the scripts.
6. Re-calibrate `UNCERTAINTY_VAR_SCALE` / `UNCERTAINTY_THRESHOLD` on your own validation data.
