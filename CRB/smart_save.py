"""Graph saving helpers for numerical records and metadata-rich images.

``plot`` and ``save_plot`` write ``.plot.json`` records for the Graph Viewer
alongside optional previews under ``Google Drive/PhD/Graphs/<system>/<plot_type>``.
The legacy ``save_plot(figure, filename)`` API saves an image with embedded
metadata under ``Google Drive/PhD/Graphs/<script_name>``.

Import these helpers from ``CRB.smart_save`` (or ``smart_save`` when running
from the CRB directory).  This module is independent of the CRB physics code
and QuTiP; Matplotlib is imported only when ``plot`` renders a figure.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
from typing import Any, Sequence
import uuid
import warnings

import numpy as np


GOOGLE_DRIVE_GRAPHS_DIRECTORY = Path("G:/My Drive/PhD/Graphs")


def _metadata_value(value: Any) -> Any:
    """Convert plot metadata to compact, JSON-safe values.

    Configuration dataclasses are recorded field-for-field.  Numerical arrays
    are summarized rather than copied into every image, while small arrays also
    retain their values so sweep grids and short result vectors remain useful.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _metadata_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, np.ndarray):
        summary: dict[str, Any] = {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "size": int(value.size),
        }
        if value.size:
            if np.issubdtype(value.dtype, np.number):
                finite = np.isfinite(value)
                summary["finite_count"] = int(np.count_nonzero(finite))
                if np.any(finite) and not np.iscomplexobj(value):
                    summary["min"] = _metadata_value(np.min(value[finite]))
                    summary["max"] = _metadata_value(np.max(value[finite]))
            if value.size <= 64:
                summary["values"] = _metadata_value(value.tolist())
        return summary
    if isinstance(value, np.generic):
        return _metadata_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        if len(items) <= 64:
            return [_metadata_value(item) for item in items]
        return {
            "type": type(value).__name__,
            "length": len(items),
            "first_values": [_metadata_value(item) for item in items[:8]],
        }
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


# ---------------------------------------------------------------------------
# Explorer plot records (.plot.json)
# ---------------------------------------------------------------------------


PLOT_SCHEMA_VERSION = 1
# Both the numerical plot-record API and the legacy image-only API share this
# destination.  Keeping the default tied to the canonical constant prevents a
# new plotting script from silently falling back to a local ``plots`` folder.
DEFAULT_PLOT_OUTPUT_DIRECTORY = GOOGLE_DRIVE_GRAPHS_DIRECTORY
SUPPORTED_AXIS_SCALES = ("linear", "log", "symlog", "logit")
SUPPORTED_COMPLEX_MODES = ("real", "imag", "abs", "phase")

_COMPLEX_MODE_HINT = "\n".join(
    ["", "Choose one of:"]
    + [f'  complex_mode="{mode}"' for mode in SUPPORTED_COMPLEX_MODES]
)
_GIT_COMMIT_CACHE: dict[str, str | None] = {}


def _plot_output_directory(output_dir: str | Path) -> Path:
    """Resolve a plot destination without allowing it outside Google Drive.

    Relative values are treated as subdirectories of ``PhD/Graphs``.  An
    absolute value is accepted only when it is already inside that tree.
    """
    requested = Path(output_dir)
    if requested.is_absolute():
        try:
            requested.relative_to(GOOGLE_DRIVE_GRAPHS_DIRECTORY)
        except ValueError as error:
            raise ValueError(
                "plot output_dir must be inside "
                f"{GOOGLE_DRIVE_GRAPHS_DIRECTORY}"
            ) from error
        return requested
    if ".." in requested.parts:
        raise ValueError("plot output_dir cannot contain '..'")
    return GOOGLE_DRIVE_GRAPHS_DIRECTORY / requested


@dataclass(frozen=True)
class PlotRecord:
    """Files written for one explorer plot record.

    The record doubles as a path to the canonical ``.plot.json`` file, so
    ``print(record)`` and ``Path(record)`` both refer to the numerical data
    rather than to the optional preview image.
    """

    id: str
    json_path: Path
    image_path: Path | None = None
    warnings: tuple[str, ...] = ()

    def __fspath__(self) -> str:
        return str(self.json_path)

    def __str__(self) -> str:
        return str(self.json_path)


def _join_parameter_path(prefix: str, key: Any) -> str:
    return f"{prefix}.{key}" if prefix else str(key)


def _non_finite_parameter(value: float) -> dict[str, str]:
    """Represent a non-finite parameter explicitly instead of ambiguously.

    Bare ``NaN``/``Infinity`` tokens are not valid JSON, and the plain strings
    ``"NaN"``/``"Infinity"`` cannot be told apart from genuine string
    parameters, so non-finite parameters are tagged instead.
    """
    if math.isnan(value):
        label = "NaN"
    else:
        label = "Infinity" if value > 0 else "-Infinity"
    return {"__nonfinite__": label}


def _parameter_value(value: Any, path: str = "") -> Any:
    """Convert a scientific parameter to JSON without discarding information.

    Unlike :func:`_metadata_value` this never summarizes: parameter arrays stay
    parameter arrays because the viewer builds its parameter space from them.
    Physical meaning is never interpreted -- keys and values are stored as
    supplied.  Values with no faithful JSON representation raise a
    :class:`TypeError` naming the offending parameter.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _parameter_value(
                getattr(value, field.name), _join_parameter_path(path, field.name)
            )
            for field in fields(value)
        }
    if isinstance(value, np.ndarray):
        return _parameter_value(value.tolist(), path)
    if isinstance(value, np.generic):
        return _parameter_value(value.item(), path)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _parameter_value(item, _join_parameter_path(path, key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            _parameter_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, complex):
        return {
            "real": _parameter_value(value.real, path),
            "imag": _parameter_value(value.imag, path),
        }
    if isinstance(value, float):
        return value if math.isfinite(value) else _non_finite_parameter(value)
    raise TypeError(
        f"Parameter {path or '<root>'!r} of type {type(value).__name__!r} has no "
        "faithful JSON representation; convert it to a number, string, sequence, "
        "or mapping before passing it as a scientific parameter."
    )


def _apply_complex_mode(values: np.ndarray, complex_mode: str) -> np.ndarray:
    if complex_mode == "real":
        return np.real(values)
    if complex_mode == "imag":
        return np.imag(values)
    if complex_mode == "abs":
        return np.abs(values)
    if complex_mode == "phase":
        return np.angle(values)
    raise ValueError(
        f"Unsupported complex_mode {complex_mode!r}; expected one of "
        f"{', '.join(SUPPORTED_COMPLEX_MODES)}"
    )


def _series_axis_values(
    values: Any,
    *,
    series_name: str,
    axis: str,
    complex_mode: str | None,
) -> tuple[np.ndarray, list[float | None], int]:
    """Return plotting values, JSON values, and the non-finite count.

    Complex data is never silently reduced: without an explicit
    ``complex_mode`` the caller is told how to choose one.  Non-finite entries
    become JSON ``null`` -- a break in the line for the viewer -- and ``nan``
    for the preview image.
    """
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(
            f"Series {series_name!r} has a {array.ndim}-dimensional {axis} array; "
            "explorer series must be one-dimensional"
        )
    if np.iscomplexobj(array):
        if complex_mode is None:
            raise ValueError(
                f"Series {series_name!r} ({axis}) contains complex values."
                + _COMPLEX_MODE_HINT
            )
        array = _apply_complex_mode(array, complex_mode)
    try:
        numeric = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"Series {series_name!r} ({axis}) holds values that are not numerical: "
            f"{error}"
        ) from error

    finite = np.isfinite(numeric)
    json_values = [
        float(item) if is_finite else None
        for item, is_finite in zip(numeric.tolist(), finite.tolist())
    ]
    plot_values = np.where(finite, numeric, np.nan)
    return plot_values, json_values, int(np.count_nonzero(~finite))


def _series_pair(name: str, entry: Any) -> tuple[Any, Any]:
    """Split an ``(x, y)`` entry supplied through ``series`` or ``data``."""
    if isinstance(entry, Mapping):
        if "x" in entry and "y" in entry:
            return entry["x"], entry["y"]
        raise ValueError(
            f"Series {name!r} was given as a mapping without 'x' and 'y' keys"
        )
    if isinstance(entry, (tuple, list)) and len(entry) == 2:
        return entry[0], entry[1]
    raise ValueError(
        f"Series {name!r} must be an (x, y) pair; got {type(entry).__name__!r}"
    )


def _normalize_series(
    x: Any,
    y: Any,
    series: Any,
    *,
    complex_mode: str | None,
    default_name: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build explorer series records from the supported input shapes.

    ``series`` supersedes ``x``/``y``.  A mapping passed as ``y`` names several
    curves that share one ``x`` array.
    """
    if series is not None:
        if not isinstance(series, Mapping):
            raise TypeError(
                "series must be a mapping of name -> (x, y); got "
                f"{type(series).__name__!r}"
            )
        if not series:
            raise ValueError("series is empty; supply at least one named curve")
        pairs = [
            (str(name), *_series_pair(str(name), entry))
            for name, entry in series.items()
        ]
    elif isinstance(y, Mapping):
        if not y:
            raise ValueError("y is an empty mapping; supply at least one named curve")
        if x is None:
            raise ValueError(
                "a shared x array is required when y names several curves; "
                "pass x=... or use series={name: (x, y)}"
            )
        pairs = [(str(name), x, values) for name, values in y.items()]
    else:
        if y is None:
            raise ValueError("nothing to plot: supply y=... or series=...")
        if x is None:
            raise ValueError(
                "an x array is required; pass x=... or use series={name: (x, y)}"
            )
        pairs = [(str(default_name), x, y)]

    records: list[dict[str, Any]] = []
    series_warnings: list[str] = []
    seen: set[str] = set()
    for name, x_values, y_values in pairs:
        if name in seen:
            raise ValueError(f"Series {name!r} is defined more than once")
        seen.add(name)

        plot_x, json_x, non_finite_x = _series_axis_values(
            x_values, series_name=name, axis="x", complex_mode=complex_mode
        )
        plot_y, json_y, non_finite_y = _series_axis_values(
            y_values, series_name=name, axis="y", complex_mode=complex_mode
        )
        if len(json_x) != len(json_y):
            raise ValueError(
                f"Series {name!r} has {len(json_x)} x values but {len(json_y)} y values"
            )
        if non_finite_x:
            series_warnings.append(
                f"Series {name} contained {non_finite_x} non-finite x values."
            )
        if non_finite_y:
            series_warnings.append(
                f"Series {name} contained {non_finite_y} non-finite values."
            )
        records.append(
            {
                "name": name,
                "x": json_x,
                "y": json_y,
                "plot_x": plot_x,
                "plot_y": plot_y,
            }
        )
    return records, series_warnings


def _caller_script_path() -> Path | None:
    """Resolve the first frame outside this module, or ``None``."""
    try:
        this_file = Path(__file__).resolve()
    except OSError:
        return None
    frame = inspect.currentframe()
    try:
        while frame is not None:
            candidate = Path(frame.f_code.co_filename)
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            if resolved != this_file:
                return resolved
            frame = frame.f_back
    finally:
        del frame
    return None


def _git_commit(directory: Path) -> str | None:
    """Best-effort short git hash; never raises and never blocks for long."""
    key = str(directory)
    if key in _GIT_COMMIT_CACHE:
        return _GIT_COMMIT_CACHE[key]
    commit: str | None = None
    try:
        completed = subprocess.run(
            ["git", "-C", key, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if completed.returncode == 0:
            commit = completed.stdout.strip() or None
    except Exception:
        commit = None
    _GIT_COMMIT_CACHE[key] = commit
    return commit


def _record_metadata(
    script_path: Path | None,
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Collect provenance.  Missing optional entries never block a save."""
    metadata: dict[str, Any] = {
        "createdAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    if script_path is not None:
        metadata["script"] = script_path.name
        metadata["scriptPath"] = str(script_path)
    try:
        metadata["pythonVersion"] = platform.python_version()
    except Exception:
        pass
    try:
        search_root = script_path.parent if script_path is not None else Path.cwd()
        commit = _git_commit(search_root)
    except Exception:
        commit = None
    if commit:
        metadata["gitCommit"] = commit
    if extra:
        metadata.update(_metadata_value(dict(extra)))
    return metadata


def _axis_description(
    label: str | None, unit: str | None, scale: str, axis: str
) -> dict[str, Any]:
    if scale not in SUPPORTED_AXIS_SCALES:
        raise ValueError(
            f"Unsupported {axis} scale {scale!r}; expected one of "
            f"{', '.join(SUPPORTED_AXIS_SCALES)}"
        )
    description: dict[str, Any] = {}
    if label is not None:
        description["label"] = str(label)
    if unit is not None:
        description["unit"] = str(unit)
    description["scale"] = scale
    return description


def _build_plot_payload(
    *,
    system: Any,
    plot_type: Any,
    params: Any,
    series_records: Sequence[Mapping[str, Any]],
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
    xunit: str | None,
    yunit: str | None,
    xscale: str,
    yscale: str,
    metadata: Mapping[str, Any] | None,
    tags: Any,
    notes: str | None,
    script_path: Path | None,
    record_warnings: Sequence[str],
) -> dict[str, Any]:
    """Validate the inputs and assemble a schema-version-1 record."""
    if not isinstance(system, str) or not system.strip():
        raise ValueError("system must be a non-empty string")
    if not isinstance(plot_type, str) or not plot_type.strip():
        raise ValueError("plot_type must be a non-empty string")
    if not isinstance(params, Mapping):
        raise TypeError(
            "params must be a mapping of scientific parameters; got "
            f"{type(params).__name__!r}"
        )
    if not series_records:
        raise ValueError("a plot record needs at least one series")

    payload: dict[str, Any] = {
        "schemaVersion": PLOT_SCHEMA_VERSION,
        "id": None,
        "system": system.strip(),
        "plotType": plot_type.strip(),
        "title": str(title) if title else plot_type.strip(),
        "parameters": {
            str(key): _parameter_value(value, str(key)) for key, value in params.items()
        },
        "axes": {
            "x": _axis_description(xlabel, xunit, xscale, "x"),
            "y": _axis_description(ylabel, yunit, yscale, "y"),
        },
        "series": [
            {"name": record["name"], "x": record["x"], "y": record["y"]}
            for record in series_records
        ],
        "metadata": _record_metadata(script_path, metadata),
    }
    if tags:
        payload["tags"] = [str(tag) for tag in tags]
    if notes:
        payload["notes"] = str(notes)
    if record_warnings:
        payload["warnings"] = list(record_warnings)
    return payload


def _sanitize_file_stem(value: str) -> str:
    """Reduce a plot type or explicit name to a filesystem-safe stem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return cleaned[:80] or "plot"


def _plot_record_output_directory(
    output_dir: str | Path,
    system: str,
    plot_type: str,
) -> Path:
    """Return ``<Google Drive root>/<system>/<plot_type>`` for a record."""
    return (
        _plot_output_directory(output_dir)
        / _sanitize_file_stem(system)
        / _sanitize_file_stem(plot_type)
    )


def _allocate_plot_paths(
    output_directory: Path, stem: str, *, reserve_image: bool
) -> tuple[str, Path, Path]:
    """Pick an unused id and file base so no existing result is overwritten."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for _ in range(64):
        plot_id = str(uuid.uuid4())
        base = f"{stem}__{timestamp}__{plot_id.replace('-', '')[:5]}"
        json_path = output_directory / f"{base}.plot.json"
        image_path = output_directory / f"{base}.png"
        if json_path.exists():
            continue
        if reserve_image and image_path.exists():
            continue
        return plot_id, json_path, image_path
    raise RuntimeError(f"could not find an unused plot file name in {output_directory}")


def _dump_plot_json(value: Any, level: int = 0, indent: int = 2) -> str:
    """Serialize a record with a readable structure but compact numerical rows.

    Standard ``indent=2`` output puts every sample of every series on its own
    line, which multiplies the size of a long trajectory.  Rows of numbers stay
    on one line here while the surrounding structure remains legible.
    """
    padding = " " * (indent * level)
    inner_padding = " " * (indent * (level + 1))
    if isinstance(value, Mapping):
        if not value:
            return "{}"
        entries = [
            f"{inner_padding}{json.dumps(str(key), ensure_ascii=False)}: "
            f"{_dump_plot_json(item, level + 1, indent)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(entries) + "\n" + padding + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(item is None or isinstance(item, (bool, int, float)) for item in value):
            return json.dumps(value, allow_nan=False)
        entries = [
            f"{inner_padding}{_dump_plot_json(item, level + 1, indent)}"
            for item in value
        ]
        return "[\n" + ",\n".join(entries) + "\n" + padding + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    """Serialize first, rename second, so an interruption leaves no half file."""
    text = _dump_plot_json(payload)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f"{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


def _preview_payload(payload: Mapping[str, Any], json_path: Path) -> dict[str, Any]:
    """Summarize a record for embedding in the preview image."""
    preview = {key: value for key, value in payload.items() if key != "series"}
    preview["record"] = json_path.name
    preview["series"] = [
        {"name": entry.get("name"), "points": len(entry.get("x", []))}
        for entry in payload.get("series", [])
    ]
    return preview


def _savefig_with_payload(
    figure: Any,
    output_path: Path,
    payload: Mapping[str, Any],
    *,
    figure_format: str,
    saved_at: str,
    **savefig_kwargs: Any,
) -> Path:
    """Save ``figure`` with the JSON payload embedded in the file metadata."""
    compact_payload = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )
    supplied_file_metadata = dict(savefig_kwargs.pop("metadata", {}) or {})
    title = supplied_file_metadata.pop("Title", output_path.stem)
    if figure_format == "png":
        savefig_kwargs["metadata"] = {
            **supplied_file_metadata,
            "Title": str(title),
            "Author": "CRB save_plot",
            "Description": compact_payload,
            "Software": "CRB.smart_save.save_plot",
            "Creation Time": saved_at,
        }
    elif figure_format == "pdf":
        savefig_kwargs["metadata"] = {
            **supplied_file_metadata,
            "Title": str(title),
            "Author": "CRB save_plot",
            "Subject": compact_payload,
            "Creator": "CRB.smart_save.save_plot",
        }
    elif figure_format == "svg":
        savefig_kwargs["metadata"] = {
            **supplied_file_metadata,
            "Title": str(title),
            "Description": compact_payload,
            "Creator": "CRB.smart_save.save_plot",
            "Date": saved_at,
        }
    elif figure_format in {"ps", "eps"}:
        savefig_kwargs["metadata"] = {
            **supplied_file_metadata,
            "Creator": "CRB.smart_save.save_plot",
        }

    figure.savefig(output_path, **savefig_kwargs)
    return output_path


def _emit_plot_record(
    payload: Mapping[str, Any],
    json_path: Path,
    image_path: Path | None,
    renderer: Any,
) -> PlotRecord:
    """Write the canonical record, then attempt the optional preview image.

    The image is produced independently: a failed render is reported but never
    invalidates the numerical record that is already on disk.
    """
    _write_json_atomically(json_path, payload)
    saved_image: Path | None = None
    if renderer is not None:
        try:
            renderer(image_path)
        except Exception as error:
            warnings.warn(
                f"Wrote {json_path.name} but the preview image failed: {error!r}",
                RuntimeWarning,
                stacklevel=3,
            )
        else:
            saved_image = image_path
    return PlotRecord(
        id=str(payload["id"]),
        json_path=json_path,
        image_path=saved_image,
        warnings=tuple(payload.get("warnings", ())),
    )


def _axis_caption(label: str | None, unit: str | None) -> str:
    if label and unit:
        return f"{label} [{unit}]"
    return str(label or unit or "")


def _series_figure_renderer(
    series_records: Sequence[Mapping[str, Any]],
    *,
    payload: Mapping[str, Any],
    json_path: Path,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
    xunit: str | None,
    yunit: str | None,
    xscale: str,
    yscale: str,
    show: bool,
) -> Any:
    """Build the preview renderer.  ``path=None`` displays without saving."""

    def render(path: Path | None) -> None:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7.0, 4.5))
        try:
            for record in series_records:
                axis.plot(record["plot_x"], record["plot_y"], label=record["name"])
            axis.set_xscale(xscale)
            axis.set_yscale(yscale)
            axis.set_xlabel(_axis_caption(xlabel, xunit))
            axis.set_ylabel(_axis_caption(ylabel, yunit))
            if title:
                axis.set_title(str(title))
            if len(series_records) > 1:
                axis.legend()
            axis.grid(True, alpha=0.3)
            figure.tight_layout()
            if path is not None:
                _savefig_with_payload(
                    figure,
                    path,
                    _preview_payload(payload, json_path),
                    figure_format="png",
                    saved_at=str(payload["metadata"]["createdAt"]),
                    dpi=150,
                )
            if show:
                plt.show()
        finally:
            plt.close(figure)

    return render


def _figure_axis_defaults(figure: Any) -> dict[str, str]:
    """Read labels, title, and scales off an existing Matplotlib figure.

    Only presentation strings are read.  Numerical data is never
    reverse-engineered from a figure; it must be supplied explicitly.
    """
    defaults: dict[str, str] = {}
    try:
        axes = list(getattr(figure, "axes", None) or [])
        if len(axes) == 1:
            axis = axes[0]
            for key, getter in (
                ("xlabel", axis.get_xlabel),
                ("ylabel", axis.get_ylabel),
                ("title", axis.get_title),
                ("xscale", axis.get_xscale),
                ("yscale", axis.get_yscale),
            ):
                value = getter()
                if isinstance(value, str) and value:
                    defaults[key] = value
        suptitle = getattr(figure, "_suptitle", None)
        if suptitle is not None:
            text = suptitle.get_text()
            if text:
                defaults["title"] = text
    except Exception:
        return {}
    return defaults


def plot(
    x: Any = None,
    y: Any = None,
    *,
    series: Mapping[str, Any] | None = None,
    system: str,
    plot_type: str,
    params: Mapping[str, Any],
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    output_dir: str | Path = DEFAULT_PLOT_OUTPUT_DIRECTORY,
    save_image: bool = True,
    show: bool = False,
    metadata: Mapping[str, Any] | None = None,
    tags: Sequence[str] | None = None,
    notes: str | None = None,
    complex_mode: str | None = None,
    script_path: str | Path | None = None,
) -> PlotRecord:
    """Save plot data beneath ``PhD/Graphs/<system>/<plot_type>``.

    The ``.plot.json`` file is the canonical artifact: keeping the arrays lets
    the viewer take differences, ratios, or maxima later without rerunning the
    simulation.  The image is only a human-readable preview.

    ``params`` describes the scientific parameter space and drives filtering in
    the viewer; ``metadata`` holds provenance and implementation details and is
    kept out of the parameter controls.  Physical meanings are never
    interpreted -- parameter keys and values are stored exactly as supplied.

    Args:
        x: Shared x values.  Ignored when ``series`` is supplied.
        y: A single y array, or a mapping of curve name to y array sharing ``x``.
        series: Mapping of curve name to an explicit ``(x, y)`` pair.  Takes
            precedence over ``x`` and ``y``.
        system: Physical system the record belongs to, e.g. ``"central_spin"``.
        plot_type: Kind of plot, e.g. ``"qfi_vs_time"``.  Sanitized before use
            in file names.
        params: Scientific parameters, stored as supplied.
        name: File-name stem.  Defaults to the sanitized ``plot_type``.
        title: Human-readable title.  Defaults to ``plot_type``.
        xlabel: Axis label recorded for x.
        ylabel: Axis label recorded for y, and the default single-series name.
        xunit: Physical unit recorded for x, e.g. ``"1/J"``.
        yunit: Physical unit recorded for y.
        xscale: One of ``linear``, ``log``, ``symlog``, ``logit``.
        yscale: One of ``linear``, ``log``, ``symlog``, ``logit``.
        output_dir: Google Drive base directory.  Outputs are placed beneath
            its ``<system>/<plot_type>`` folders.  Absolute paths must already
            be within ``PhD/Graphs``.
        save_image: Whether to render the optional preview image.
        show: Whether to display the preview figure.
        metadata: Extra provenance, e.g. solver settings or tolerances.
        tags: Free-form labels stored with the record.
        notes: Free-form note stored with the record.
        complex_mode: One of ``real``, ``imag``, ``abs``, ``phase``.  Required
            when any series holds complex values.
        script_path: Source script recorded as provenance.  Defaults to the
            caller's file.

    Returns:
        A :class:`PlotRecord` naming the written files.  It also works directly
        as a path to the ``.plot.json`` file.

    Raises:
        ValueError: If validation fails, or if a series holds complex values
            and ``complex_mode`` was not chosen.
    """
    if complex_mode is not None and complex_mode not in SUPPORTED_COMPLEX_MODES:
        raise ValueError(
            f"Unsupported complex_mode {complex_mode!r}; expected one of "
            f"{', '.join(SUPPORTED_COMPLEX_MODES)}"
        )

    series_records, record_warnings = _normalize_series(
        x,
        y,
        series,
        complex_mode=complex_mode,
        default_name=ylabel or plot_type,
    )
    resolved_script = (
        Path(script_path).resolve()
        if script_path is not None
        else _caller_script_path()
    )
    payload = _build_plot_payload(
        system=system,
        plot_type=plot_type,
        params=params,
        series_records=series_records,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xunit=xunit,
        yunit=yunit,
        xscale=xscale,
        yscale=yscale,
        metadata=metadata,
        tags=tags,
        notes=notes,
        script_path=resolved_script,
        record_warnings=record_warnings,
    )

    output_directory = _plot_record_output_directory(
        output_dir,
        system,
        plot_type,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    plot_id, json_path, image_path = _allocate_plot_paths(
        output_directory,
        _sanitize_file_stem(name or plot_type),
        reserve_image=save_image,
    )
    payload["id"] = plot_id

    renderer = None
    if save_image or show:
        renderer = _series_figure_renderer(
            series_records,
            payload=payload,
            json_path=json_path,
            title=payload["title"],
            xlabel=xlabel,
            ylabel=ylabel,
            xunit=xunit,
            yunit=yunit,
            xscale=xscale,
            yscale=yscale,
            show=show,
        )
    return _emit_plot_record(
        payload, json_path, image_path if save_image else None, renderer
    )


def capture_params(*names: str, depth: int = 1) -> dict[str, Any]:
    """Collect named variables from the calling scope into a ``params`` dict.

    Preferred over ``params=locals()``, which sweeps up unrelated objects and
    can serialize very large arrays by accident.

    Args:
        *names: Variable names to read from the caller's locals, then globals.
        depth: How many frames up to look.  ``1`` is the direct caller.

    Returns:
        A plain dict mapping each name to its current value.

    Raises:
        NameError: If a requested name is not defined in that scope.
    """
    frame = inspect.currentframe()
    try:
        target = frame.f_back if frame is not None else None
        for _ in range(max(0, depth - 1)):
            target = target.f_back if target is not None else None
        if target is None:
            raise RuntimeError("capture_params could not inspect the calling scope")
        captured: dict[str, Any] = {}
        for name in names:
            if name in target.f_locals:
                captured[name] = target.f_locals[name]
            elif name in target.f_globals:
                captured[name] = target.f_globals[name]
            else:
                raise NameError(
                    f"capture_params could not find {name!r} in the calling scope"
                )
        return captured
    finally:
        del frame


def save_plot(
    figure: Any,
    filename: str | Path | None = None,
    *,
    system: str | None = None,
    plot_type: str | None = None,
    params: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | Sequence[Any] | None = None,
    name: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xunit: str | None = None,
    yunit: str | None = None,
    xscale: str | None = None,
    yscale: str | None = None,
    output_dir: str | Path = DEFAULT_PLOT_OUTPUT_DIRECTORY,
    save_image: bool = True,
    tags: Sequence[str] | None = None,
    notes: str | None = None,
    complex_mode: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    script_path: str | Path | None = None,
    **savefig_kwargs: Any,
) -> PlotRecord | Path:
    """Save an existing Matplotlib figure, in explorer or legacy mode.

    Explorer mode -- selected by passing ``system``, ``plot_type``, ``params``,
    and ``data`` -- writes a ``.plot.json`` record next to the figure so the
    Graph Viewer can reload the numbers::

        fig, ax = plt.subplots()
        ax.plot(t, qfi)
        ax.set_xlabel("t")

        save_plot(
            fig,
            system="central_spin",
            plot_type="qfi_vs_time",
            params=params,
            data={"QFI": (t, qfi)},
        )

    The numerical ``data`` is required: arrays are never reverse-engineered
    from a figure.  Axis labels, title, and scales are read off a single-axes
    figure when not given explicitly.

    Legacy mode -- selected by passing ``filename`` and no explorer arguments
    -- keeps the original behavior: the figure is written to
    ``My Drive/PhD/Graphs/<script_name>/<filename>`` with the whole metadata
    payload embedded in the image file, and the saved :class:`Path` is
    returned.

    Args:
        figure: A Matplotlib-compatible figure exposing ``savefig``.
        filename: Legacy mode: the desired file name, whose directory
            components are ignored because the destination is centrally
            managed.  Explorer mode: an optional file-name stem.
        system: Explorer mode: physical system the record belongs to.
        plot_type: Explorer mode: kind of plot, sanitized for file names.
        params: Explorer mode: scientific parameters, stored as supplied.
        data: Explorer mode: the numerical curves, as a mapping of name to an
            ``(x, y)`` pair, or a single ``(x, y)`` pair.
        name: Explorer mode: file-name stem, overriding ``filename``.
        title: Recorded title.  Defaults to the figure's title.
        xlabel: Recorded x label.  Defaults to the figure's x label.
        ylabel: Recorded y label.  Defaults to the figure's y label.
        xunit: Physical unit recorded for x.
        yunit: Physical unit recorded for y.
        xscale: Recorded x scale.  Defaults to the figure's x scale.
        yscale: Recorded y scale.  Defaults to the figure's y scale.
        output_dir: Explorer mode: Google Drive base directory.  Outputs are
            placed beneath its ``<system>/<plot_type>`` folders.  Absolute
            paths must already be within ``PhD/Graphs``.
        save_image: Explorer mode: whether to save the figure alongside the
            record.
        tags: Free-form labels stored with the record.
        notes: Free-form note stored with the record.
        complex_mode: One of ``real``, ``imag``, ``abs``, ``phase``.  Required
            when any curve holds complex values.
        metadata: Legacy mode: the configuration, inputs, and derived values to
            preserve.  Explorer mode: execution provenance only, kept apart
            from ``params``.
        script_path: Source script.  Defaults to the caller's file.
        **savefig_kwargs: Keyword arguments forwarded to ``figure.savefig``.

    Returns:
        Explorer mode: a :class:`PlotRecord`.  Legacy mode: the absolute
        :class:`Path` of the saved figure.
    """
    explorer_arguments = {
        "system": system,
        "plot_type": plot_type,
        "params": params,
        "data": data,
    }
    resolved_script = (
        Path(script_path).resolve()
        if script_path is not None
        else _caller_script_path()
    )
    if not any(value is not None for value in explorer_arguments.values()):
        if filename is None:
            raise TypeError(
                "save_plot needs either a filename (legacy mode) or "
                "system/plot_type/params/data (explorer mode)"
            )
        return _save_plot_legacy(
            figure,
            filename,
            metadata=metadata,
            script_path=resolved_script,
            **savefig_kwargs,
        )

    missing = [key for key, value in explorer_arguments.items() if value is None]
    if missing:
        raise TypeError(
            "explorer mode needs system, plot_type, params, and data; missing "
            + ", ".join(missing)
        )
    if complex_mode is not None and complex_mode not in SUPPORTED_COMPLEX_MODES:
        raise ValueError(
            f"Unsupported complex_mode {complex_mode!r}; expected one of "
            f"{', '.join(SUPPORTED_COMPLEX_MODES)}"
        )

    figure_defaults = _figure_axis_defaults(figure)
    resolved_xscale = xscale or figure_defaults.get("xscale") or "linear"
    resolved_yscale = yscale or figure_defaults.get("yscale") or "linear"
    resolved_xlabel = xlabel if xlabel is not None else figure_defaults.get("xlabel")
    resolved_ylabel = ylabel if ylabel is not None else figure_defaults.get("ylabel")
    resolved_title = title if title is not None else figure_defaults.get("title")

    curves = data if isinstance(data, Mapping) else {resolved_ylabel or plot_type: data}
    series_records, record_warnings = _normalize_series(
        None,
        None,
        curves,
        complex_mode=complex_mode,
        default_name=resolved_ylabel or plot_type,
    )
    payload = _build_plot_payload(
        system=system,
        plot_type=plot_type,
        params=params,
        series_records=series_records,
        title=resolved_title,
        xlabel=resolved_xlabel,
        ylabel=resolved_ylabel,
        xunit=xunit,
        yunit=yunit,
        xscale=resolved_xscale,
        yscale=resolved_yscale,
        metadata=metadata,
        tags=tags,
        notes=notes,
        script_path=resolved_script,
        record_warnings=record_warnings,
    )

    output_directory = _plot_record_output_directory(
        output_dir,
        system,
        plot_type,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = name or (Path(filename).stem if filename is not None else None) or plot_type
    plot_id, json_path, image_path = _allocate_plot_paths(
        output_directory, _sanitize_file_stem(stem), reserve_image=save_image
    )
    payload["id"] = plot_id

    renderer = None
    if save_image:

        def renderer(path: Path) -> None:
            _savefig_with_payload(
                figure,
                path,
                _preview_payload(payload, json_path),
                figure_format="png",
                saved_at=str(payload["metadata"]["createdAt"]),
                **savefig_kwargs,
            )

    return _emit_plot_record(
        payload, json_path, image_path if save_image else None, renderer
    )


def _save_plot_legacy(
    figure: Any,
    filename: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    script_path: Path | None = None,
    **savefig_kwargs: Any,
) -> Path:
    """Save a figure under ``My Drive/PhD/Graphs/<script_name>`` with metadata.

    ``metadata`` should contain the configuration and any plot-specific
    derived quantities.  The complete JSON payload is embedded directly in
    PNG, PDF, and SVG outputs, so the plot remains self-contained when copied
    or shared.  Formats with limited metadata support receive a creator tag.
    """
    source_script = Path(script_path or _caller_script_path() or __file__).resolve()
    script_name = source_script.stem

    requested_name = Path(filename).name
    if not requested_name:
        raise ValueError("filename must include a file name")
    output_directory = GOOGLE_DRIVE_GRAPHS_DIRECTORY / script_name
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / requested_name

    explicit_format = savefig_kwargs.get("format")
    figure_format = str(
        explicit_format or output_path.suffix.lstrip(".") or "png"
    ).lower()
    if not output_path.suffix:
        output_path = output_path.with_suffix(f".{figure_format}")

    saved_at = datetime.now(timezone.utc).isoformat()
    figure_size = None
    if hasattr(figure, "get_size_inches"):
        figure_size = [float(item) for item in figure.get_size_inches()]
    payload = {
        "schema_version": 1,
        "saved_at_utc": saved_at,
        "source_script": str(source_script),
        "script_name": script_name,
        "figure": {
            "filename": output_path.name,
            "format": figure_format,
            "dpi": _metadata_value(
                savefig_kwargs.get("dpi", getattr(figure, "dpi", None))
            ),
            "size_inches": figure_size,
        },
        "parameters": _metadata_value(metadata or {}),
    }
    return _savefig_with_payload(
        figure,
        output_path,
        payload,
        figure_format=figure_format,
        saved_at=saved_at,
        **savefig_kwargs,
    )


__all__ = [
    "DEFAULT_PLOT_OUTPUT_DIRECTORY",
    "GOOGLE_DRIVE_GRAPHS_DIRECTORY",
    "PLOT_SCHEMA_VERSION",
    "PlotRecord",
    "SUPPORTED_AXIS_SCALES",
    "SUPPORTED_COMPLEX_MODES",
    "capture_params",
    "plot",
    "save_plot",
]
