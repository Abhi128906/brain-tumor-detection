import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi.testclient import TestClient
from backend.app.main import app
import numpy as np
import cv2

class NeuroVisionAPITestCase(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("models_configured", data)

    def test_list_models(self):
        res = self.client.get("/api/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("models", data)
        self.assertGreaterEqual(len(data["models"]), 2)

    def test_demo_scan(self):
        res = self.client.get("/api/demo/glioma")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["sample_id"], "glioma")
        self.assertTrue(data["image_base64"].startswith("data:image/"))

    def test_analyze_pipeline(self):
        sample_path = os.path.join(BASE_DIR, 'app', 'static', 'samples', 'sample_glioma.jpg')
        if not os.path.exists(sample_path):
            self.skipTest("Sample file missing")

        with open(sample_path, 'rb') as f:
            files = {'file': ('sample_glioma.jpg', f, 'image/jpeg')}
            data = {'model_choice': 'nnunet-2d'}
            res = self.client.post("/api/analyze", files=files, data=data)

        self.assertEqual(res.status_code, 200)
        resp = res.json()
        self.assertIn("result_id", resp)
        self.assertIn("classification", resp)
        self.assertIn("segmentation", resp)
        self.assertIn("quality", resp)
        self.assertIn("mesh_3d", resp)
        self.assertIn("human_review", resp)
        self.assertIn("uncertainty", resp)
        self.assertIn("consistency", resp)
        self.assertIn("is_available", resp["classification"])
        self.assertTrue(resp["mesh_3d"]["has_mesh"])

    def test_classification_never_depends_on_filename(self):
        """Regression: the old pipeline returned 'Meningioma 94%' for any file named *meningioma*."""
        sample_path = os.path.join(BASE_DIR, 'app', 'static', 'samples', 'sample_glioma.jpg')
        if not os.path.exists(sample_path):
            self.skipTest("Sample file missing")
        raw = open(sample_path, 'rb').read()
        outs = []
        for name in ("sample_glioma.jpg", "my_meningioma_case.jpg", "scan_001.jpg"):
            res = self.client.post("/api/analyze", files={'file': (name, raw, 'image/jpeg')},
                                   data={'model_choice': 'nnunet-2d'})
            self.assertEqual(res.status_code, 200)
            c = res.json()["classification"]
            outs.append((c["is_available"], c["predicted_class"], c["confidence"],
                         [p["probability"] for p in c["probabilities"]]))
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[0], outs[2])
        if not outs[0][0]:                      # no checkpoint => nothing may be invented
            self.assertEqual(outs[0][3], [])
            self.assertEqual(outs[0][2], 0.0)

    def test_invalid_model_choice_rejected(self):
        res = self.client.post("/api/analyze", files={'file': ('a.jpg', b'x', 'image/jpeg')}, data={'model_choice': 'nope'})
        self.assertEqual(res.status_code, 400)

    def test_benchmark_endpoint(self):
        res = self.client.post("/api/benchmark")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("models", data)
        self.assertIn("speedup", data)
        for m in data["models"]:                # measured or null - never hardcoded
            self.assertIn("weights_trained", m)
            self.assertTrue(m["val_dice_score"] is None or 0.0 <= m["val_dice_score"] <= 1.0)

if __name__ == "__main__":
    unittest.main()
