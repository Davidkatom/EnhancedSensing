"""Regression checks for script folders and independent graph exports.

Run with: python -m unittest discover -s CRB/tests -v
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from CRB import smart_save


class PlotExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(plt.close, "all")

    def read(self, record):
        return json.loads(record.json_path.read_text(encoding="utf-8"))

    def test_plot_uses_script_folders_and_preserves_comparison_curves(self):
        for script in ("experiment_a.py", "experiment_b.py"):
            record = smart_save.plot(
                [1, 2], {"driven": [3, 4], "reference": [1, 1]},
                system="test", plot_type="qfi", params={"N": 2},
                output_dir=self.root / "plot", script_path=script, save_image=False,
            )
            self.assertEqual(record.json_path.parent, self.root / "plot" / Path(script).stem)
            self.assertEqual(len(self.read(record)["series"]), 2)
        self.assertEqual(smart_save.DEFAULT_PLOT_OUTPUT_DIRECTORY, Path("plot"))

    def test_caller_inference(self):
        record = smart_save.plot([1], [2], system="test", plot_type="qfi", params={},
                           output_dir=self.root, save_image=False)
        self.assertEqual(record.json_path.parent.name, Path(__file__).stem)

    def test_panels_have_independent_arrays_scales_and_previews(self):
        figure, (first, second) = plt.subplots(2, 1, sharex=True)
        first.plot([1, 2], [3, 4], label="QFI")
        first.plot([1, 2], [2, 2], label="reference")
        second.plot([1, 2], [-1, 1], label="spin")
        first.set_ylabel("QFI")
        first.set_yscale("log")
        second.set(xlabel="Time", ylabel="Spin")
        original_visibility = [tick.label1.get_visible() for tick in first.xaxis.get_major_ticks()]
        records = smart_save.save_plot(
            figure, system="test", plot_type="experiment", params={"N": 2},
            panels={
                "qfi": {"axis": first, "data": {"QFI": ([1, 2], [3, 4]),
                                                   "reference": ([1, 2], [2, 2])}},
                "spin": {"axis": second, "data": {"spin": ([1, 2], [-1, 1])}},
            }, output_dir=self.root, script_path="experiment.py", dpi=60,
        )
        qfi, spin = map(self.read, records)
        self.assertEqual([s["name"] for s in qfi["series"]], ["QFI", "reference"])
        self.assertEqual(spin["series"][0]["y"], [-1.0, 1.0])
        self.assertEqual(qfi["axes"]["y"]["scale"], "log")
        self.assertEqual(spin["axes"]["y"]["scale"], "linear")
        self.assertEqual(qfi["axes"]["x"]["label"], "Time")
        self.assertEqual(qfi["parameters"], spin["parameters"])
        self.assertEqual(qfi["metadata"]["experimentId"], spin["metadata"]["experimentId"])
        self.assertNotEqual(qfi["plotType"], spin["plotType"])
        self.assertNotEqual(records[0].id, records[1].id)
        for record in records:
            self.assertTrue(record.image_path.is_file())
            with Image.open(record.image_path) as image:
                self.assertLess(image.height, figure.get_figheight() * 60)
        self.assertEqual(figure.axes, [first, second])
        self.assertTrue(plt.fignum_exists(figure.number))
        self.assertEqual(original_visibility, [tick.label1.get_visible() for tick in first.xaxis.get_major_ticks()])

    def test_unassigned_subplots_fail_before_writing(self):
        figure, axes = plt.subplots(2)
        for axis in axes:
            axis.plot([1, 2], [3, 4])
        with self.assertRaisesRegex(ValueError, "subplots needs panels"):
            smart_save.save_plot(figure, system="test", plot_type="mixed", params={},
                           data={"mixed": ([1, 2], [3, 4])}, output_dir=self.root)
        self.assertFalse(list(self.root.rglob("*.plot.json")))

    def test_colorbar_belongs_to_heatmap(self):
        figure, axis = plt.subplots()
        image = axis.imshow([[1, 2], [3, 4]])
        figure.colorbar(image, ax=axis)
        axis.set(xlabel="x", ylabel="y")
        record = smart_save.save_plot(
            figure, system="test", plot_type="map", params={},
            data={"row 0": ([0, 1], [1, 2]), "row 1": ([0, 1], [3, 4])},
            output_dir=self.root, script_path="heatmap.py", dpi=50,
        )
        self.assertIsInstance(record, smart_save.PlotRecord)
        self.assertEqual(self.read(record)["axes"]["x"]["label"], "x")
        self.assertTrue(record.image_path.is_file())

    def test_legacy_panels_and_colorbars_save_locally(self):
        figure, (heatmap, curve) = plt.subplots(1, 2)
        image = heatmap.imshow([[1, 2], [3, 4]])
        figure.colorbar(image, ax=heatmap)
        curve.plot([1, 2], [3, 4])
        paths = smart_save.save_plot(figure, "old.png", script_path="legacy.py",
                               output_dir=self.root, dpi=60)
        self.assertEqual(len(paths), 2)
        self.assertEqual(len(figure.axes), 3)
        for path in paths:
            self.assertEqual(path.parent, self.root / "legacy")
            self.assertTrue(path.is_file())
            with Image.open(path) as image:
                self.assertLess(image.width, figure.get_figwidth() * 60)

    def test_real_phase_cycling_exports_signal_and_spectrum(self):
        from CRB.OrderAnalysis import plot_phase_cycling as phase
        cfg = phase.PhaseCyclingConfig(N=2, interrogation_time=0.1,
                                       n_phase_samples=9, show_figure=False, figure_dpi=50)
        result = phase.run_phase_cycling(cfg)
        with patch.object(phase, "save_plot", side_effect=lambda *args, **kw:
                          smart_save.save_plot(*args, **kw, output_dir=self.root)):
            records = phase.plot_phase_cycling(result, cfg)
        self.assertEqual(len(records), 2)
        signal, spectrum = map(self.read, records)
        self.assertEqual([len(s["x"]) for s in signal["series"]], [9])
        self.assertEqual([len(s["x"]) for s in spectrum["series"]], [5, 5, 5])
        self.assertEqual(signal["axes"]["x"]["unit"], "rad")
        self.assertNotEqual(signal["axes"], spectrum["axes"])
        for record in records:
            self.assertTrue(record.image_path.is_file())


if __name__ == "__main__":
    unittest.main()
