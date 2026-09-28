"""Checks for the graph-saving API extracted from crb_core.

Run with: python -m unittest discover -s CRB/tests -p test_smart_save.py -v
"""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from CRB import smart_save


class SmartSaveTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        destination = patch.object(smart_save, "GOOGLE_DRIVE_GRAPHS_DIRECTORY", self.root)
        destination.start()
        self.addCleanup(destination.stop)
        self.addCleanup(plt.close, "all")

    def test_plot_preserves_data_parameters_and_caller(self):
        coupling = 1.5
        record = smart_save.plot(
            np.array([0, 1]), {"signal": [2, 3], "reference": [1, 1]},
            system="central_spin", plot_type="qfi", save_image=False,
            params=smart_save.capture_params("coupling"), output_dir="run",
        )
        self.assertEqual(record.json_path.parent, self.root / "run" / "central_spin" / "qfi")
        self.assertEqual(Path(record), record.json_path)
        self.assertIsNone(record.image_path)
        payload = json.loads(record.json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["parameters"], {"coupling": coupling})
        self.assertEqual(payload["series"], [
            {"name": "signal", "x": [0.0, 1.0], "y": [2.0, 3.0]},
            {"name": "reference", "x": [0.0, 1.0], "y": [1.0, 1.0]},
        ])
        self.assertEqual(Path(payload["metadata"]["scriptPath"]), Path(__file__).resolve())

    def test_save_plot_preserves_preview_and_axis_metadata(self):
        figure, axis = plt.subplots()
        axis.plot([1, 2], [3, 4])
        axis.set(xlabel="Time", ylabel="QFI", yscale="log")
        record = smart_save.save_plot(
            figure, system="central_spin", plot_type="qfi", params={"N": 2},
            data={"QFI": ([1, 2], [3, 4])}, output_dir=self.root, dpi=50,
        )
        payload = json.loads(record.json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["axes"]["x"]["label"], "Time")
        self.assertEqual(payload["axes"]["y"]["scale"], "log")
        with Image.open(record.image_path) as image:
            self.assertEqual(image.info["Software"], "CRB.smart_save.save_plot")
            self.assertEqual(json.loads(image.info["Description"])["record"], record.json_path.name)
        self.assertTrue(plt.fignum_exists(figure.number))

    def test_legacy_save_preserves_script_folder_and_embedded_metadata(self):
        figure, axis = plt.subplots()
        axis.plot([1, 2], [3, 4])
        path = smart_save.save_plot(figure, "legacy.png", metadata={"N": 2}, dpi=50)
        self.assertEqual(path, self.root / Path(__file__).stem / "legacy.png")
        with Image.open(path) as image:
            payload = json.loads(image.info["Description"])
        self.assertEqual(payload["parameters"], {"N": 2})
        self.assertEqual(Path(payload["source_script"]), Path(__file__).resolve())

    def test_imports_without_physics_or_matplotlib(self):
        repository = Path(__file__).resolve().parents[2]
        for directory, module in ((repository, "CRB.smart_save"), (repository / "CRB", "smart_save")):
            with self.subTest(module=module):
                completed = subprocess.run(
                    [sys.executable, "-c", f"""
import importlib
import sys
class BlockPhysicsAndPlotting:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {{'CRB.crb_core', 'crb_core', 'qutip', 'matplotlib'}}:
            raise ImportError('Unexpected dependency: ' + fullname)
sys.meta_path.insert(0, BlockPhysicsAndPlotting())
helper = importlib.import_module({module!r})
assert callable(helper.save_plot)
assert callable(helper.plot)
"""], cwd=directory, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
