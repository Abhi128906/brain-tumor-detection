import os
import sys
import time
import uuid
import statistics
from collections import OrderedDict
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Response
from fastapi.responses import JSONResponse

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from common.preprocessing import CLASS_NAMES, segmenter_tensor  # noqa: E402
from common.inference import segmenter_prob_tta  # noqa: E402

from backend.app import config  # noqa: E402
from backend.app.models.segmenter import (  # noqa: E402
    get_segmenter, nnUNet2D, ModelUnavailableError, segmenter_weights_available)
from backend.app.models.lightweight_segmenter import (  # noqa: E402
    get_lightweight_segmenter, FastBrainUNet, lightweight_weights_available)
from backend.app.models.classifier import (  # noqa: E402
    get_classifier_bundle, classify_image, resolve_device, CLASS_DISPLAY, TUMOR_DETAILS)
from backend.app.services.preprocessing import (  # noqa: E402
    decode_image_bytes, make_clinical_overlay, make_transparent_mask, numpy_to_base64)
from backend.app.services.quality_assessment import evaluate_scan_quality  # noqa: E402
from backend.app.services.measurements import compute_measurements  # noqa: E402
from backend.app.services.reconstruction import generate_3d_mesh  # noqa: E402
from backend.app.services.postprocess import postprocess_mask  # noqa: E402
from backend.app.services.uncertainty import estimate_mc_uncertainty  # noqa: E402
from backend.app.services.metrics_store import get_model_metrics, load_classifier_metadata  # noqa: E402

router = APIRouter(prefix="/api")

# Bounded in-memory store for recent analysis results (oldest evicted first)
RESULTS_STORE: "OrderedDict[str, dict]" = OrderedDict()

BASE_DIR = config.BASE_DIR
SAMPLES_DIR = os.path.join(BASE_DIR, 'app', 'static', 'samples')
VALID_MODELS = ("nnunet-2d", "fast-brain-unet")


def _store_result(result_id: str, payload: dict):
    RESULTS_STORE[result_id] = payload
    while len(RESULTS_STORE) > config.MAX_STORED_RESULTS:
        RESULTS_STORE.popitem(last=False)


@lru_cache(maxsize=None)
def _arch_param_count(kind: str) -> int:
    m = nnUNet2D() if kind == "nnunet" else FastBrainUNet()
    return int(sum(p.numel() for p in m.parameters()))


def _fmt_params(n: int) -> str:
    return f"{n / 1e6:.2f}M"


# --------------------------------------------------------------------------- health / models
@router.get("/health")
def health_check():
    has_gpu = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_gpu else "CPU"
    clf = get_classifier_bundle()
    return {
        "status": "healthy",
        "service": "NeuroVision AI Medical Engine",
        "version": "2.5.0",
        "device": device_name,
        "cuda_available": has_gpu,
        "models_configured": {
            "nnunet_segmenter": segmenter_weights_available(),
            "classifier": clf.is_configured,            # True only if a checkpoint actually LOADED
            "lightweight_unet": lightweight_weights_available(),
        },
        "classifier_error": clf.error,
    }


@router.get("/models")
def list_models():
    clf = get_classifier_bundle()
    meta = load_classifier_metadata()
    nn_m = get_model_metrics("nnunet-2d")
    fb_m = get_model_metrics("fast-brain-unet")
    nn_n, fb_n = _arch_param_count("nnunet"), _arch_param_count("fast")
    return {
        "models": [
            {
                "id": "nnunet-2d",
                "name": "nnU-Net Medical Segmentation (Baseline)",
                "task": "2D/Volumetric Lesion Segmentation",
                "parameters": _fmt_params(nn_n),
                "architecture": "U-Net with InstanceNorm & LeakyReLU",
                "weights_status": "Loaded (trained)" if segmenter_weights_available() else "Weights not found",
                "evaluated": nn_m is not None,
                "input_resolution": "224x224x3",
            },
            {
                "id": "fast-brain-unet",
                "name": "FastBrain-UNet (Lightweight Edge Model)",
                "task": "Low-Latency Lesion Segmentation",
                "parameters": f"{_fmt_params(fb_n)} ({nn_n / fb_n:.1f}x smaller)",
                "architecture": "Depthwise Separable U-Net",
                "weights_status": "Loaded (trained)" if lightweight_weights_available() else "Not trained yet",
                "evaluated": fb_m is not None,
                "input_resolution": "224x224x3"
            },
            {
                "id": "classifier",
                "name": f"{clf.backbone or 'Tumour'} Multi-Class Classifier",
                "task": "4-Class Tumor Typing",
                "classes": [CLASS_DISPLAY[c] for c in CLASS_NAMES],
                "parameters": _fmt_params(sum(p.numel() for p in clf.model.parameters())) if clf.model else "n/a",
                "weights_status": "Loaded (trained, calibrated)" if clf.is_configured else "Unavailable - no valid checkpoint",
                "test_accuracy": meta.get("test_acc") if clf.is_configured else None,
                "temperature": clf.temperature if clf.is_configured else None,
            }
        ]
    }


@router.get("/demo/{sample_id}")
def load_demo_scan(sample_id: str):
    """
    Returns pre-configured clinical demonstration MRI scans with ground truth info.
    """
    sample_files = {
        "glioma": ("sample_glioma.jpg", "Glioma"),
        "meningioma": ("sample_meningioma.jpg", "Meningioma"),
        "pituitary": ("sample_pituitary.jpg", "Pituitary Tumor"),
        "no_tumor": ("sample_no_tumor.jpg", "No Tumor (Healthy)")
    }

    if sample_id not in sample_files:
        raise HTTPException(status_code=404, detail="Demo sample not found. Choose: glioma, meningioma, pituitary, no_tumor")

    filename, class_name = sample_files[sample_id]
    filepath = os.path.join(SAMPLES_DIR, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"File {filename} missing from samples directory.")

    with open(filepath, "rb") as f:
        file_bytes = f.read()

    img_rgb, is_vol, metadata, _ = decode_image_bytes(file_bytes, filename)
    img_b64 = numpy_to_base64(img_rgb)

    return {
        "sample_id": sample_id,
        "filename": filename,
        "demonstration_class": class_name,
        "image_base64": img_b64,
        "metadata": metadata
    }


@router.post("/upload")
async def upload_scan(file: UploadFile = File(...), slice_idx: Optional[int] = Form(None)):
    """
    Validates uploaded MRI file, checks dimensions, formats, and returns preview thumbnail and metadata.
    """
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(file_bytes) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds {config.MAX_UPLOAD_MB}MB safety limit.")

    try:
        img_rgb, is_volumetric, metadata, _ = decode_image_bytes(file_bytes, file.filename, slice_idx=slice_idx)
        preview_b64 = numpy_to_base64(img_rgb)

        return {
            "filename": file.filename,
            "size_bytes": len(file_bytes),
            "is_volumetric": is_volumetric,
            "metadata": metadata,
            "preview_base64": preview_b64
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# --------------------------------------------------------------------------- analysis
def _build_classification(probs, bundle, class_ms):
    """Build the classification block. NEVER fabricates numbers: no checkpoint => explicit 'unavailable'."""
    if probs is None:
        return {
            "is_available": False,
            "abstained": False,
            "predicted_class": "unclassified",
            "display_name": "Classification unavailable",
            "confidence": 0.0,
            "probabilities": [],
            "severity": "Not assessed",
            "description": ("No trained classifier checkpoint could be loaded, so tumour type was NOT assessed. "
                            "Only the segmentation findings below are model outputs. "
                            + (f"Reason: {bundle.error}" if bundle.error else "")),
            "clinical_actions": "Tumour typing requires radiologist assessment (or a trained classifier: run training/train.py).",
            "inference_time_ms": class_ms,
            "is_trained_model": False,
            "status_label": "Classifier not installed",
            "temperature": None,
            "tta": False,
        }

    top_idx = int(np.argmax(probs))
    top_id = CLASS_NAMES[top_idx]
    top_conf = float(probs[top_idx])
    abstained = top_conf < config.ABSTAIN_THRESHOLD
    info = TUMOR_DETAILS.get(top_id, TUMOR_DETAILS['no_tumor'])

    block = {
        "is_available": True,
        "abstained": abstained,
        "predicted_class": top_id,
        "display_name": (f"Uncertain (leading: {CLASS_DISPLAY[top_id]})" if abstained else CLASS_DISPLAY[top_id]),
        "confidence": round(top_conf * 100.0, 1),
        "probabilities": [
            {"class_id": c, "display_name": CLASS_DISPLAY[c],
             "probability": round(float(probs[i]), 4), "percentage": round(float(probs[i]) * 100.0, 1)}
            for i, c in enumerate(CLASS_NAMES)
        ],
        "severity": "Not determined - specialist review required" if abstained else info['severity'],
        "description": (f"Calibrated confidence ({round(top_conf * 100, 1)}%) is below the {round(config.ABSTAIN_THRESHOLD * 100)}% "
                        f"abstain threshold, so no tumour type is asserted." if abstained else info['description']),
        "clinical_actions": ("Radiologist review of the full study before any typing." if abstained else info['clinical_actions']),
        "inference_time_ms": class_ms,
        "is_trained_model": True,
        "status_label": f"Trained {bundle.backbone} (calibrated, T={bundle.temperature:.2f})",
        "temperature": round(bundle.temperature, 4),
        "tta": config.USE_TTA,
    }
    return block


@router.post("/analyze")
async def run_analysis(
    file: UploadFile = File(...),
    slice_idx: Optional[int] = Form(None),
    model_choice: str = Form("nnunet-2d")
):
    """
    Multi-stage AI pipeline:
    1. Decode + shared preprocessing (identical to training)
    2. Segmentation (nnU-Net / FastBrain-UNet) with flip-TTA + connected-component clean-up
    3. Calibrated 4-class classification with TTA, or an explicit "unavailable" state (never invented)
    4. Monte-Carlo-dropout uncertainty on the segmentation
    5. Quality assessment, measurements, 3D mesh
    6. Cross-model consistency checks + human-review recommendation
    """
    if model_choice not in VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"model_choice must be one of {VALID_MODELS}")

    pipeline_t0 = time.time()
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(file_bytes) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds {config.MAX_UPLOAD_MB}MB safety limit.")

    try:
        img_rgb, is_volumetric, metadata, raw_vol = decode_image_bytes(file_bytes, file.filename, slice_idx=slice_idx)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    h_orig, w_orig = img_rgb.shape[:2]

    device = resolve_device()

    # ---- 1. Segmentation --------------------------------------------------------------
    try:
        if model_choice == "fast-brain-unet":
            seg_model, _ = get_lightweight_segmenter(device=device)
        else:
            seg_model, _ = get_segmenter(device=device)
    except ModelUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))

    seg_x = segmenter_tensor(img_rgb).to(device)
    seg_prob_224 = segmenter_prob_tta(seg_model, seg_x, tta=config.USE_TTA)
    seg_prob_orig = cv2.resize(seg_prob_224, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    min_area = max(10, int(round(0.0004 * h_orig * w_orig)))
    bin_mask_orig = postprocess_mask(seg_prob_orig, config.SEG_THRESHOLD, min_area_px=min_area)

    lesion_fraction = float(bin_mask_orig.sum()) / float(bin_mask_orig.size)
    has_lesion = lesion_fraction >= config.LESION_MIN_FRACTION

    # ---- 2. Classification (real or explicitly unavailable) ---------------------------------
    class_t0 = time.time()
    bundle = get_classifier_bundle(device)
    probs = classify_image(img_rgb, device)
    class_ms = round((time.time() - class_t0) * 1000, 2)
    classification = _build_classification(probs, bundle, class_ms)

    cls_id = classification["predicted_class"] if (classification["is_available"] and not classification["abstained"]) else None
    top_conf = classification["confidence"] / 100.0

    # ---- 3. MC-dropout uncertainty -------------------------------------------------------
    uncertainty = None
    try:
        unc = estimate_mc_uncertainty(seg_model, seg_x, device=device, target_shape=(h_orig, w_orig), original_image=img_rgb)
        if unc is None:
            uncertainty = {"available": False, "reason": "Selected model has no dropout layers; uncertainty cannot be estimated."}
        else:
            uncertainty = {k: v for k, v in unc.items() if k not in ("mean_prediction", "normalized_map")}
    except Exception as e:  # uncertainty must never take the whole analysis down
        uncertainty = {"available": False, "reason": f"Uncertainty estimation failed: {e}"}

    # ---- 4. Quality, measurements, 3D --------------------------------------------------------
    quality_data = evaluate_scan_quality(img_rgb, binary_mask=bin_mask_orig, filename=file.filename)

    voxel_spacing = metadata.get("voxel_spacing")
    measurements = compute_measurements(bin_mask_orig, pixel_spacing=voxel_spacing[:2] if voxel_spacing else None,
                                        is_volumetric=is_volumetric)

    slice_info = {
        "current_slice": metadata.get("current_slice", 0),
        "total_slices": metadata.get("total_slices", 1)
    }
    # If the classifier says "no tumour" but a lesion was segmented, do not let the mesh code suppress it:
    # the mask is shown, the disagreement is flagged below.
    mesh_class = None if (cls_id == 'no_tumor' and has_lesion) else cls_id
    mesh_data = generate_3d_mesh(
        bin_mask_orig,
        is_volumetric=is_volumetric,
        voxel_spacing=voxel_spacing if voxel_spacing else (1.0, 1.0, 1.0),
        raw_image=img_rgb,
        predicted_class=mesh_class,
        slice_info=slice_info
    )

    overlay_img = make_clinical_overlay(img_rgb, bin_mask_orig)
    transparent_mask_img = make_transparent_mask(bin_mask_orig)
    orig_b64 = numpy_to_base64(img_rgb)
    mask_b64 = numpy_to_base64(bin_mask_orig * 255)
    overlay_b64 = numpy_to_base64(overlay_img)
    transparent_b64 = numpy_to_base64(transparent_mask_img)

    # ---- 5. Cross-model consistency + human review ------------------------------------------------
    review_reasons = []
    if not classification["is_available"]:
        consistency = {"status": "not_assessed", "detail": "Classifier unavailable; segmentation-only result."}
        review_reasons.append("Tumour typing unavailable (no trained classifier checkpoint). Segmentation output only.")
    elif classification["abstained"]:
        consistency = {"status": "not_assessed", "detail": "Classifier abstained (low calibrated confidence)."}
        review_reasons.append(f"Classifier abstained: calibrated confidence {classification['confidence']}% "
                              f"< {round(config.ABSTAIN_THRESHOLD * 100)}% threshold.")
    else:
        if top_conf < 0.65:
            review_reasons.append(f"Classification confidence is below standard threshold ({classification['confidence']}% < 65%).")
        if cls_id == 'no_tumor' and has_lesion:
            consistency = {"status": "conflict",
                           "detail": f"Classifier: no tumour; segmentation: lesion covering {lesion_fraction * 100:.2f}% of the slice."}
            review_reasons.append("Conflict: classifier reports no tumour but the segmentation model found a lesion-sized "
                                  f"region ({lesion_fraction * 100:.2f}% of the slice). Both outputs are shown unmodified.")
        elif cls_id not in (None, 'no_tumor') and not has_lesion:
            consistency = {"status": "conflict",
                           "detail": f"Classifier: {CLASS_DISPLAY[cls_id]}; segmentation found no lesion."}
            review_reasons.append(f"Conflict: classifier predicts {CLASS_DISPLAY[cls_id]} but no lesion was segmented.")
        else:
            consistency = {"status": "consistent", "detail": "Classifier and segmentation agree."}

    if quality_data['overall_status'] == "Needs review":
        review_reasons.extend(quality_data['warnings'])
    if uncertainty and uncertainty.get("review_recommended"):
        review_reasons.append(f"High boundary uncertainty ({uncertainty['boundary_uncertainty']}% > "
                              f"{uncertainty['threshold_pct']}%): the tumour margin is not reliably delineated.")

    requires_human = len(review_reasons) > 0
    review_status = ("Analysis completed — review recommended" if requires_human
                     else "Analysis completed — no automated warning detected")

    total_pipeline_time_ms = round((time.time() - pipeline_t0) * 1000, 1)
    result_id = str(uuid.uuid4())[:8]

    _store_result(result_id, {
        "filename": file.filename,
        "metadata": metadata,
        "classification": {k: classification[k] for k in
                           ("is_available", "abstained", "predicted_class", "display_name", "confidence", "probabilities")},
        "segmentation": measurements,
        "uncertainty": {k: v for k, v in (uncertainty or {}).items() if not str(k).endswith("base64")},
        "consistency": consistency,
        "quality": quality_data,
        "mesh": mesh_data,
        "human_review": {"status": review_status, "reasons": review_reasons},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

    return {
        "result_id": result_id,
        "filename": file.filename,
        "model_used": "nnU-Net (Baseline)" if model_choice == "nnunet-2d" else "FastBrain-UNet (Lightweight)",
        "is_volumetric": is_volumetric,
        "metadata": metadata,
        "original_image_base64": orig_b64,
        "mask_base64": mask_b64,
        "overlay_base64": overlay_b64,
        "transparent_overlay_base64": transparent_b64,
        "classification": classification,
        "segmentation": measurements,
        "segmentation_meta": {
            "threshold": config.SEG_THRESHOLD,
            "tta": config.USE_TTA,
            "postprocessed": True,
            "lesion_fraction_pct": round(lesion_fraction * 100, 3),
        },
        "uncertainty": uncertainty,
        "consistency": consistency,
        "quality": quality_data,
        "mesh_3d": {
            "has_mesh": mesh_data['has_mesh'],
            "vertex_count": mesh_data.get('vertex_count', 0),
            "face_count": mesh_data.get('face_count', 0),
            "vertices": mesh_data.get('vertices', []),
            "faces": mesh_data.get('faces', []),
            "is_genuine_3d": mesh_data.get('is_genuine_3d', False),
            "limitation_note": mesh_data.get('limitation_note', ''),
            "anatomical_location": mesh_data.get('anatomical_location', ''),
            "hemisphere": mesh_data.get('hemisphere', ''),
            "centroid_3d": mesh_data.get('centroid_3d', [0.0, 0.0, 0.0])
        },
        "human_review": {
            "review_status": review_status,
            "reasons": review_reasons,
            "requires_human_attention": requires_human,
            "disclaimer": "This is a research prototype. NeuroVision AI outputs do not constitute a confirmed medical diagnosis and must be evaluated by a qualified clinician."
        },
        "total_pipeline_time_ms": total_pipeline_time_ms
    }


# --------------------------------------------------------------------------- benchmark
def _model_or_arch(getter, device):
    """Trained model if available, else the bare architecture (latency of an architecture is still valid)."""
    try:
        return getter(device=device)[0], True
    except ModelUnavailableError:
        return getter(device=device, require_weights=False)[0], False


def _measure_latency_ms(model, device, runs=15, warmup=3):
    model.eval()
    dummy = torch.randn(1, 3, 224, 224, device=device)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        for _ in range(runs):
            t0 = time.perf_counter()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return round(statistics.median(times), 1)


@router.post("/benchmark")
def benchmark_models():
    """
    Measured latency / parameter count / parameter memory, plus Dice & IoU read from
    models/segmentation_metrics.json (written by training/train_segmentation.py).
    Nothing here is hardcoded; unevaluated metrics are reported as null.
    """
    device = resolve_device()
    specs = [
        ("nnunet-2d", "nnU-Net 2D (Baseline)", get_segmenter, "Clinical Workstation / High-Accuracy Cloud"),
        ("fast-brain-unet", "FastBrain-UNet (Lightweight)", get_lightweight_segmenter, "Mobile MRI Units / Edge Radiologist Tablets"),
    ]
    rows = []
    for model_id, name, getter, deployment in specs:
        model, trained = _model_or_arch(getter, device)
        n_params = int(sum(p.numel() for p in model.parameters()))
        param_mb = round(sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6, 2)
        m = get_model_metrics(model_id) if trained else None
        rows.append({
            "id": model_id,
            "name": name,
            "parameters": _fmt_params(n_params),
            "param_count": n_params,
            "latency_ms": _measure_latency_ms(model, device),
            "memory_mb": param_mb,                      # parameter memory (measured), not process RSS
            "weights_trained": trained,
            "val_dice_score": m.get("dice") if m else None,
            "val_iou": m.get("iou") if m else None,
            "dice_tumour_slices": m.get("dice_tumour_only") if m else None,
            "hd95_px": m.get("hd95_px") if m else None,
            "metrics_source": ("models/segmentation_metrics.json" if m else
                               ("not evaluated - run training/train_segmentation.py --eval-only" if trained else "weights not trained")),
            "intended_deployment": deployment,
        })

    base, fast = rows[0], rows[1]
    for r in rows:
        r["compression_ratio"] = "1.0x (Reference)" if r is base else f"{base['param_count'] / r['param_count']:.1f}x smaller"

    return {
        "models": rows,
        "device_used": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "speedup": round(base["latency_ms"] / max(fast["latency_ms"], 0.1), 1),
    }


# --------------------------------------------------------------------------- downloads
@router.get("/results/{result_id}/mesh/{mesh_format}")
def download_mesh(result_id: str, mesh_format: str):
    """
    Downloads exported 3D mesh in STL or OBJ format.
    """
    if result_id not in RESULTS_STORE:
        raise HTTPException(status_code=404, detail="Analysis result not found or expired.")

    mesh = RESULTS_STORE[result_id]['mesh']
    if not mesh.get('has_mesh'):
        raise HTTPException(status_code=400, detail="No 3D mesh was generated for this scan.")

    if mesh_format.lower() == "stl":
        return Response(
            content=mesh['stl_ascii'],
            media_type="application/sla",
            headers={"Content-Disposition": f"attachment; filename=neurovision_mesh_{result_id}.stl"}
        )
    elif mesh_format.lower() == "obj":
        return Response(
            content=mesh['obj_ascii'],
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename=neurovision_mesh_{result_id}.obj"}
        )
    else:
        raise HTTPException(status_code=400, detail="Supported mesh formats: stl, obj")


@router.get("/results/{result_id}/export/{export_format}")
def export_summary(result_id: str, export_format: str):
    """
    Exports summary report in JSON or CSV.
    """
    if result_id not in RESULTS_STORE:
        raise HTTPException(status_code=404, detail="Analysis result not found.")

    res = RESULTS_STORE[result_id]

    if export_format.lower() == "json":
        clean_res = {k: v for k, v in res.items() if k != 'mesh'}
        return JSONResponse(clean_res)

    elif export_format.lower() == "csv":
        cls = res['classification']
        unc = res.get('uncertainty') or {}
        csv_rows = [
            "Metric,Value",
            f"Result ID,{result_id}",
            f"Timestamp,{res['timestamp']}",
            f"Classifier Available,{cls['is_available']}",
            f"Predicted Class,{cls['display_name']}",
            f"Classifier Abstained,{cls['abstained']}",
            f"Confidence,{cls['confidence']}%",
            f"Lesion Detected,{res['segmentation']['has_lesion']}",
            f"Pixel Count,{res['segmentation']['pixel_count']}",
            f"Occupied Area Percentage,{res['segmentation']['percentage_occupied']}%",
            f"Boundary Uncertainty,{unc.get('boundary_uncertainty', 'n/a')}",
            f"Segmentation/Classifier Consistency,{res['consistency']['status']}",
            f"Quality Status,{res['quality']['overall_status']}",
            f"Human Review Status,{res['human_review']['status']}"
        ]
        return Response(
            content="\n".join(csv_rows),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=neurovision_analysis_{result_id}.csv"}
        )

    else:
        raise HTTPException(status_code=400, detail="Supported export formats: json, csv")
