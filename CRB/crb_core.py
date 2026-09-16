"""Shared numerical utilities for collective-bath CRB analyses.

The exact solver in this module uses the symmetric bath subspace and the
Hamiltonian convention

    H = Omega_0 * sigma_x + J * sigma_z * S_z + omega * S_x,

where ``S_{x,z} = 2 * jmat(N / 2, "x,z")``.  The bath drive ``omega`` defaults
to zero for backward compatibility.  Keeping that convention explicit is
important when comparing these helpers with scripts that use a factor of one
half in either the drive or collective-spin operators.

The module also provides the explorer plotting helpers :func:`plot` and
:func:`save_plot`, which write ``.plot.json`` records -- the canonical
numerical artifact read by the Graph Viewer -- next to an optional preview
image under ``Google Drive/PhD/Graphs/<system>/<plot_type>``.
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
import qutip as qt


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
            "Software": "CRB.crb_core.save_plot",
            "Creation Time": saved_at,
        }
    elif figure_format == "pdf":
        savefig_kwargs["metadata"] = {
            **supplied_file_metadata,
            "Title": str(title),
            "Author": "CRB save_plot",
            "Subject": compact_payload,
            "Creator": "CRB.crb_core.save_plot",
        }
    elif figure_format == "svg":
        savefig_kwargs["metadata"] = {
            **supplied_file_metadata,
            "Title": str(title),
            "Description": compact_payload,
            "Creator": "CRB.crb_core.save_plot",
            "Date": saved_at,
        }
    elif figure_format in {"ps", "eps"}:
        savefig_kwargs["metadata"] = {
            **supplied_file_metadata,
            "Creator": "CRB.crb_core.save_plot",
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


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration fields shared by the collective-bath analyses."""

    N: int = 8
    J_nominal: float = 1.0
    dJ: float = 1e-3

    t_min: float = 0.01
    t_max: float = 60.0
    n_steps: int = 300

    gamma: float = 0.0
    beta: float = 0.0

    qfi_tol: float = 1e-12
    qcrb_eps: float = 1e-15
    t_overhead: float = 5.0


# ---------------------------------------------------------------------------
# Operator and state construction
# ---------------------------------------------------------------------------


def build_spin_operators(N: int) -> dict[str, object]:
    """Build central-spin and collective-bath operators.

    The bath is represented in its spin-``N / 2`` symmetric subspace, whose
    dimension is ``N + 1``.
    """
    if N < 0:
        raise ValueError("N must be non-negative")

    S_spin = N / 2.0
    dim_bath = N + 1
    Jx = 2.0 * qt.jmat(S_spin, "x")
    Jz = 2.0 * qt.jmat(S_spin, "z")
    I_bath = qt.qeye(dim_bath)

    sx = qt.sigmax()
    sz = qt.sigmaz()
    si = qt.qeye(2)

    return {
        "S_spin": S_spin,
        "dim_bath": dim_bath,
        "Jx": Jx,
        "Jz": Jz,
        "I_bath": I_bath,
        "sx": sx,
        "sz": sz,
        "si": si,
        "sx_s": qt.tensor(sx, I_bath),
        "sz_s": qt.tensor(sz, I_bath),
        "Sx_op": qt.tensor(si, Jx),
        "Sz_op": qt.tensor(si, Jz),
    }


def central_spin_state(theta: float, phi: float = 0.0) -> qt.Qobj:
    """Return a central-spin pure state at Bloch angles ``theta`` and ``phi``.

    The convention is ``cos(theta/2)|0> + exp(i phi) sin(theta/2)|1>``.
    Consequently, ``theta = 0`` gives ``|0>`` and ``theta = pi/2, phi = 0``
    gives the ``|+x>`` state used by the legacy analyses.
    """
    return (
        np.cos(theta / 2.0) * qt.basis(2, 0)
        + np.exp(1j * phi) * np.sin(theta / 2.0) * qt.basis(2, 1)
    ).unit()


def build_initial_state(
    S_spin: float,
    central_theta: float = np.pi / 2.0,
) -> qt.Qobj:
    """Return a configurable central spin and bath ``+x`` product state."""
    initial_central_state = central_spin_state(central_theta)
    plus_state_bath = qt.spin_coherent(S_spin, np.pi / 2.0, 0.0)
    return qt.tensor(initial_central_state, plus_state_bath)


def build_hamiltonian(
    Omega_0: float,
    J: float,
    N: int,
    omega: float = 0.0,
) -> qt.Qobj:
    """Return ``Omega_0 sigma_x + J sigma_z S_z + omega S_x``."""
    operators = build_spin_operators(N)
    return (
        Omega_0 * operators["sx_s"]
        + J * operators["sz_s"] * operators["Sz_op"]
        + omega * operators["Sx_op"]
    )


def optimal_sz2_bath_state(N: int) -> np.ndarray:
    """Return the optimal probe state for the coefficient of an ``S_z^2`` generator.

    Following arXiv:0710.0285 (Boixo et al.), the optimal initial state for
    estimating ``gamma`` in ``exp(-i gamma t h)`` is the equal superposition of
    the eigenstates of ``h`` with the largest and smallest eigenvalues.  For
    ``h = S_z^2`` these are ``|m = N/2>`` and the state closest to ``m = 0``
    (exactly ``m = 0`` for even ``N``).
    """
    if N < 1:
        raise ValueError("N must be positive")

    s_vals = 2.0 * np.real(np.diag(qt.jmat(N / 2.0, "z").full()))
    idx_max = int(np.argmax(s_vals**2))
    idx_min = int(np.argmin(s_vals**2))
    state = np.zeros(N + 1, dtype=complex)
    state[[idx_max, idx_min]] = 1.0 / np.sqrt(2.0)
    return state


def coherent_bath_state(N: int, theta: float, phi: float = 0.0) -> np.ndarray:
    """Return the spin-``N/2`` coherent state at polar angle ``theta`` from +z.

    This is the product-state probe parametrized by the paper's angle ``beta``
    (arXiv:0710.0285), where the coherent state is produced from the north-pole
    state ``|m = N/2>`` by a rotation ``theta`` about ``y``.  ``theta = pi/2`` is
    the equatorial ``+x`` state used as the default elsewhere; ``theta -> 0``
    approaches the ``|m = N/2>`` eigenstate of ``S_z``.
    """
    if N < 1:
        raise ValueError("N must be positive")

    return qt.spin_coherent(N / 2.0, theta, phi).full().ravel().astype(complex)


def build_bath_operators(N: int) -> dict[str, np.ndarray]:
    """Return dense collective bath operators using the solver's scaling."""
    if N < 0:
        raise ValueError("N must be non-negative")

    spin = N / 2.0
    return {
        "I": np.eye(N + 1, dtype=complex),
        "Jx": 2.0 * qt.jmat(spin, "x").full(),
        "Jy": 2.0 * qt.jmat(spin, "y").full(),
        "Jz": 2.0 * qt.jmat(spin, "z").full(),
    }


# ---------------------------------------------------------------------------
# Exact block-decomposed bath evolution
# ---------------------------------------------------------------------------


def evolve_bath_density_matrix_noiseless(
    Omega_0: float,
    J: float,
    time: float,
    N: int,
    bath_state: np.ndarray,
    omega: float = 0.0,
    central_theta: float = np.pi / 2.0,
) -> np.ndarray:
    """Return the reduced bath state after exact noiseless evolution.

    The time-independent joint Hamiltonian is propagated spectrally in the
    central-spin times symmetric-bath space.  This is especially efficient
    when only one interrogation time is required.
    """
    if N < 1:
        raise ValueError("N must be positive")
    if time < 0.0:
        raise ValueError("time must be non-negative")

    dim_bath = N + 1
    bath_vector = np.asarray(bath_state, dtype=complex).ravel()
    if bath_vector.shape != (dim_bath,):
        raise ValueError(
            f"bath_state must have dimension N + 1 = {dim_bath}, "
            f"got {bath_vector.shape[0]}"
        )

    central_vector = central_spin_state(central_theta).full().ravel()
    initial_state = np.kron(central_vector, bath_vector)
    hamiltonian = build_hamiltonian(
        Omega_0=Omega_0,
        omega=omega,
        J=J,
        N=N,
    ).full()
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    eigenbasis_amplitudes = eigenvectors.conj().T @ initial_state
    evolved_state = eigenvectors @ (
        np.exp(-1j * eigenvalues * time) * eigenbasis_amplitudes
    )

    amplitudes = evolved_state.reshape(2, dim_bath)
    return amplitudes.T @ amplitudes.conj()


def get_bath_density_matrices(
    Omega_0: float,
    J: float,
    tlist: Sequence[float] | np.ndarray,
    N: int = 10,
    gamma: float = 1.0,
    beta: float = 1.0,
    bath_state: np.ndarray | None = None,
    omega: float = 0.0,
    central_theta: float = np.pi / 2.0,
) -> list[np.ndarray]:
    """Evolve and return ``rho_B(t) = Tr_central[rho(t)]``.

    With ``omega = 0``, each bath coherence closes on a four-dimensional
    central-spin block.  The resulting independent constant-coefficient
    Liouvillians are solved by eigendecomposition.  A nonzero ``omega * S_x``
    couples those blocks, so the evolution is performed in the full
    central-spin times symmetric-bath space instead.

    The initial central-spin state is
    ``cos(central_theta/2)|0> + sin(central_theta/2)|1>``.  Its default is the
    legacy ``|+x>`` state.  ``bath_state`` defaults to the ``+x`` spin coherent
    state used by the legacy analyses.
    """
    if N < 0:
        raise ValueError("N must be non-negative")

    times = np.asarray(tlist, dtype=float)
    if times.ndim != 1:
        raise ValueError("tlist must be one-dimensional")

    S_spin = N / 2.0
    dim_bath = N + 1

    # Ordering matches qt.jmat and qt.spin_coherent.
    s_vals = 2.0 * np.real(np.diag(qt.jmat(S_spin, "z").full()))
    if bath_state is None:
        chi = qt.spin_coherent(S_spin, np.pi / 2.0, 0.0).full().ravel()
    else:
        chi = np.asarray(bath_state, dtype=complex).ravel()
        if chi.shape[0] != dim_bath:
            raise ValueError(
                f"bath_state must have dimension N + 1 = {dim_bath}, "
                f"got {chi.shape[0]}"
            )

    if omega != 0.0:
        if np.any(times < 0.0) or np.any(np.diff(times) < 0.0):
            raise ValueError(
                "tlist must be non-negative and increasing when omega is nonzero"
            )
        if len(times) == 0:
            return []

        operators = build_spin_operators(N)
        hamiltonian = build_hamiltonian(
            Omega_0=Omega_0,
            J=J,
            N=N,
            omega=omega,
        )
        initial_central_state = central_spin_state(central_theta)
        bath_ket = qt.Qobj(chi, dims=[[dim_bath], [1]])
        initial_state = qt.tensor(initial_central_state, bath_ket)

        collapse_operators = []
        if beta > 0.0:
            collapse_operators.append(np.sqrt(beta) * operators["sx_s"])
        if gamma > 0.0:
            collapse_operators.append(np.sqrt(gamma) * operators["sz_s"])

        # QuTiP treats the first entry of tlist as the initial time.  Prepending
        # zero preserves the API's convention that every requested time is
        # measured from the supplied initial state, even when tlist starts later.
        prepend_zero = times[0] > 0.0
        solver_times = (
            np.concatenate(([0.0], times)) if prepend_zero else times
        )
        result = qt.mesolve(
            hamiltonian,
            initial_state,
            solver_times,
            c_ops=collapse_operators,
            e_ops=[],
        )
        states = result.states[1:] if prepend_zero else result.states
        return [state.ptrace(1).full() for state in states]

    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    identity_2 = np.eye(2, dtype=complex)
    identity_4 = np.eye(4, dtype=complex)

    # Row-major vectorization: vec(A R B) = (A kron B.T) vec(R).
    central_state = central_spin_state(central_theta).full().ravel()
    central_rho_0 = np.outer(central_state, central_state.conj()).reshape(4)
    dissipator = (
        beta * (np.kron(sx, sx.T) - identity_4)
        + gamma * (np.kron(sz, sz.T) - identity_4)
    )

    bath = np.zeros((len(times), dim_bath, dim_bath), dtype=complex)
    for i in range(dim_bath):
        H_i = Omega_0 * sx + J * s_vals[i] * sz
        for j in range(i, dim_bath):
            H_j = Omega_0 * sx + J * s_vals[j] * sz
            liouvillian = (
                -1j
                * (
                    np.kron(H_i, identity_2)
                    - np.kron(identity_2, H_j.T)
                )
                + dissipator
            )

            eigenvalues, eigenvectors = np.linalg.eig(liouvillian)
            coefficients = np.linalg.solve(eigenvectors, central_rho_0)
            coefficients *= chi[i] * np.conj(chi[j])
            evolved = eigenvectors @ (
                np.exp(np.outer(eigenvalues, times)) * coefficients[:, None]
            )

            reduced_element = evolved[0] + evolved[3]
            bath[:, i, j] = reduced_element
            if j != i:
                bath[:, j, i] = np.conj(reduced_element)

    return list(bath)


# ---------------------------------------------------------------------------
# Quantum Fisher information
# ---------------------------------------------------------------------------


def qfi_from_rho_and_drho(
    rho: np.ndarray,
    drho: np.ndarray,
    tol: float = 1e-12,
) -> tuple[float, np.ndarray]:
    """Return the mixed-state QFI and symmetric logarithmic derivative."""
    rho = np.asarray(rho, dtype=complex)
    drho = np.asarray(drho, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square matrix")
    if drho.shape != rho.shape:
        raise ValueError("drho must have the same shape as rho")

    rho = 0.5 * (rho + rho.conj().T)
    drho = 0.5 * (drho + drho.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    eigenvalues = np.real(eigenvalues)
    eigenvalues[np.abs(eigenvalues) < tol] = 0.0

    derivative_eigenbasis = eigenvectors.conj().T @ drho @ eigenvectors
    denominator = eigenvalues[:, None] + eigenvalues[None, :]
    inverse_weight = np.zeros_like(denominator)
    valid = denominator > tol
    inverse_weight[valid] = 2.0 / denominator[valid]

    qfi = np.sum(np.abs(derivative_eigenbasis) ** 2 * inverse_weight)
    sld_eigenbasis = derivative_eigenbasis * inverse_weight
    sld = eigenvectors @ sld_eigenbasis @ eigenvectors.conj().T
    sld = 0.5 * (sld + sld.conj().T)
    return float(np.real(qfi)), sld


def qfi_vectorized(
    rho: np.ndarray,
    drho: np.ndarray,
    tol: float = 1e-12,
) -> float:
    """Return only the scalar QFI, without retaining the SLD."""
    rho = np.asarray(rho, dtype=complex)
    drho = np.asarray(drho, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square matrix")
    if drho.shape != rho.shape:
        raise ValueError("drho must have the same shape as rho")

    rho = 0.5 * (rho + rho.conj().T)
    drho = 0.5 * (drho + drho.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    # Preserve qfi_N_scaling.py's treatment of numerical noise by clipping
    # all negative eigenvalues in the scalar-only implementation.
    eigenvalues = np.clip(np.real(eigenvalues), 0.0, None)

    derivative_eigenbasis = eigenvectors.conj().T @ drho @ eigenvectors
    denominator = eigenvalues[:, None] + eigenvalues[None, :]
    valid = denominator > tol
    return float(
        np.real(
            2.0
            * np.sum(
                np.abs(derivative_eigenbasis[valid]) ** 2 / denominator[valid]
            )
        )
    )


def compute_bath_qfi_trajectory(
    bath_rhos_plus: Sequence[np.ndarray],
    bath_rhos_minus: Sequence[np.ndarray],
    dJ: float,
    tol: float = 1e-12,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Compute a finite-difference bath QFI trajectory and its intermediates."""
    if len(bath_rhos_plus) != len(bath_rhos_minus):
        raise ValueError("plus and minus trajectories must have the same length")
    if dJ == 0:
        raise ValueError("dJ must be non-zero")

    qfi_t = np.zeros(len(bath_rhos_plus))
    sld_t: list[np.ndarray] = []
    rho_t: list[np.ndarray] = []
    drho_t: list[np.ndarray] = []
    for index, (rho_plus, rho_minus) in enumerate(
        zip(bath_rhos_plus, bath_rhos_minus)
    ):
        rho = 0.5 * (rho_plus + rho_minus)
        drho = (rho_plus - rho_minus) / (2.0 * dJ)
        qfi_t[index], sld = qfi_from_rho_and_drho(rho, drho, tol=tol)
        sld_t.append(sld)
        rho_t.append(rho)
        drho_t.append(drho)

    return qfi_t, sld_t, rho_t, drho_t


def observable_moment_fisher(
    rho: np.ndarray,
    drho: np.ndarray,
    observable: np.ndarray,
    var_floor: float = 1e-12,
) -> float:
    """Classical Fisher information for estimating a parameter from the *mean*
    of a single observable ``A`` (method of moments / error propagation):

        F_cl = (d<A>/dtheta)^2 / Var(A),
        <A> = Tr[rho A],   Var(A) = Tr[rho A^2] - <A>^2,   d<A> = Tr[drho A].

    The associated classical CRB is ``1/sqrt(F_cl) = sqrt(Var A)/|d<A>|`` -- the
    precision achievable by reading out ``<A>`` alone.  It satisfies
    ``F_cl <= F_Q``, so this bound never beats the QFI.  ``var_floor`` guards the
    division when the state is (near) an eigenstate of ``A``.
    """
    A = np.asarray(observable, dtype=complex)
    rho = 0.5 * (rho + rho.conj().T)
    drho = 0.5 * (drho + drho.conj().T)
    mean = float(np.real(np.trace(rho @ A)))
    variance = float(np.real(np.trace(rho @ (A @ A)))) - mean * mean
    variance = max(variance, var_floor)
    derivative_of_mean = float(np.real(np.trace(drho @ A)))
    return derivative_of_mean * derivative_of_mean / variance


def observable_projective_fisher(
    rho: np.ndarray,
    drho: np.ndarray,
    observable: np.ndarray,
    tol: float = 1e-12,
) -> float:
    """Classical Fisher information of a *projective* measurement of ``A``.

    Diagonalizing ``A = sum_k a_k |k><k|``, the outcome probabilities are
    ``p_k = <k|rho|k>`` with derivatives ``dp_k = <k|drho|k>``, giving

        F_cl = sum_{k : p_k > tol} (dp_k)^2 / p_k.

    Unlike :func:`observable_moment_fisher`, this uses the full outcome
    distribution (all moments of ``A``), not just the mean, so it captures
    parameter dependence hidden in the variance.  It still obeys
    ``F_cl <= F_Q``.
    """
    A = np.asarray(observable, dtype=complex)
    A = 0.5 * (A + A.conj().T)
    rho = 0.5 * (rho + rho.conj().T)
    drho = 0.5 * (drho + drho.conj().T)
    _, eigenvectors = np.linalg.eigh(A)
    probabilities = np.real(np.diag(eigenvectors.conj().T @ rho @ eigenvectors))
    derivatives = np.real(np.diag(eigenvectors.conj().T @ drho @ eigenvectors))
    valid = probabilities > tol
    return float(np.sum(derivatives[valid] ** 2 / probabilities[valid]))


def observable_projective_score(
    rho: np.ndarray,
    drho: np.ndarray,
    observable: np.ndarray,
    tol: float = 1e-12,
) -> tuple[float, np.ndarray]:
    """Return full projective FI and its outcome-score operator.

    In an eigenbasis ``|k>`` of ``A``, define ``p_k = <k|rho|k>`` and
    ``dp_k = <k|drho|k>``.  The score operator for the full projective
    measurement is

        L_A = sum_{k : p_k > tol} (dp_k / p_k) |k><k|.

    Its SLD-weighted squared norm is the full outcome-distribution FI,

        (L_A, L_A)_rho = Tr(rho L_A^2)
                         = sum_k (dp_k)^2 / p_k = F_C^proj(A).

    For a nondegenerate observable, ``L_A`` is the orthogonal projection of
    the SLD onto the commutative measurement algebra generated by ``A``.
    """
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    A = np.asarray(observable, dtype=complex)
    rho = np.asarray(rho, dtype=complex)
    drho = np.asarray(drho, dtype=complex)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("observable must be a square matrix")
    if rho.shape != A.shape or drho.shape != A.shape:
        raise ValueError("rho, drho, and observable must have the same shape")
    A = 0.5 * (A + A.conj().T)
    rho = 0.5 * (rho + rho.conj().T)
    drho = 0.5 * (drho + drho.conj().T)
    _, eigenvectors = np.linalg.eigh(A)
    probabilities = np.real(np.diag(eigenvectors.conj().T @ rho @ eigenvectors))
    derivatives = np.real(np.diag(eigenvectors.conj().T @ drho @ eigenvectors))
    valid = probabilities > tol
    scores = np.zeros_like(probabilities)
    scores[valid] = derivatives[valid] / probabilities[valid]
    score_operator = eigenvectors @ np.diag(scores) @ eigenvectors.conj().T
    score_operator = 0.5 * (score_operator + score_operator.conj().T)
    fisher = float(np.sum(derivatives[valid] ** 2 / probabilities[valid]))
    return fisher, score_operator


# ---------------------------------------------------------------------------
# QCRB utilities
# ---------------------------------------------------------------------------


def compute_qcrb_matrices(
    qfi_matrix: np.ndarray,
    tlist: Sequence[float] | np.ndarray,
    t_overhead: float,
    qcrb_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized/unnormalized QCRBs and normalized optimum indices."""
    qfi = np.asarray(qfi_matrix, dtype=float)
    times = np.asarray(tlist, dtype=float)
    if qfi.ndim != 2:
        raise ValueError("qfi_matrix must be two-dimensional")
    if times.ndim != 1 or qfi.shape[1] != len(times):
        raise ValueError("the second qfi_matrix axis must match tlist")

    normalized = np.sqrt((times + t_overhead)[None, :] / (qfi + qcrb_eps))
    unnormalized = 1.0 / np.sqrt(qfi + qcrb_eps)
    optimal_time_indices = np.argmin(normalized, axis=1)
    return normalized, unnormalized, optimal_time_indices


# ---------------------------------------------------------------------------
# Fisher-metric projection onto operator bases
# ---------------------------------------------------------------------------


def frobenius_orthonormal_span(
    operators: Sequence[np.ndarray],
    tol: float = 1e-12,
) -> list[np.ndarray]:
    """Return a stable Hermitian basis for the same real operator span."""
    basis: list[np.ndarray] = []
    for operator in operators:
        candidate = np.asarray(operator, dtype=complex)
        candidate = 0.5 * (candidate + candidate.conj().T)
        for previous in basis:
            candidate -= np.real(np.vdot(previous, candidate)) * previous
        norm = np.linalg.norm(candidate, ord="fro")
        if norm > tol:
            basis.append(candidate / norm)
    return basis


def fisher_metric_projection(
    rho: np.ndarray,
    drho: np.ndarray,
    operators: Sequence[np.ndarray],
    ridge: float = 1e-10,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Project the SLD onto an operator span using Fisher normal equations."""
    stable_basis = frobenius_orthonormal_span(operators)
    count = len(stable_basis)
    if count == 0:
        return 0.0, np.empty(0), np.empty((0, 0)), np.empty(0)

    rho = 0.5 * (rho + rho.conj().T)
    drho = 0.5 * (drho + drho.conj().T)
    metric = np.empty((count, count), dtype=float)
    derivative = np.empty(count, dtype=float)
    rho_times_operators = [rho @ operator for operator in stable_basis]

    for i, operator_i in enumerate(stable_basis):
        derivative[i] = float(np.real(np.trace(operator_i @ drho)))
        for j in range(i, count):
            value = float(
                np.real(np.trace(rho_times_operators[i] @ stable_basis[j]))
            )
            metric[i, j] = value
            metric[j, i] = value

    metric = 0.5 * (metric + metric.T)
    scale = max(
        float(np.max(np.abs(np.diag(metric)))),
        np.finfo(float).tiny,
    )
    regularized_metric = metric + ridge * scale * np.eye(count)
    try:
        coefficients = np.linalg.solve(regularized_metric, derivative)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(
            regularized_metric, derivative, rcond=None
        )[0]

    projected_qfi = max(float(np.dot(coefficients, derivative)), 0.0)
    return projected_qfi, coefficients, metric, derivative


def fisher_metric_decomposition(
    rho: np.ndarray,
    drho: np.ndarray,
    operators: Sequence[np.ndarray],
    rtol: float = 1e-10,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decompose the SLD in a supplied Hermitian operator basis.

    Unlike :func:`fisher_metric_projection`, this function preserves the
    caller's basis and therefore returns coefficients attached directly to the
    supplied operators.  It solves the Fisher-metric normal equations using a
    relative-eigenvalue pseudoinverse,

        G_ij = (A_i, A_j)_rho,
        b_i = (A_i, L)_rho = Tr(A_i drho),
        G c = b.

    The reconstructed operator is ``sum_i c_i A_i`` and the captured Fisher
    information is ``c dot b``.
    """
    if rtol <= 0.0:
        raise ValueError("rtol must be positive")
    rho = np.asarray(rho, dtype=complex)
    drho = np.asarray(drho, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square matrix")
    if drho.shape != rho.shape:
        raise ValueError("drho must have the same shape as rho")
    hermitian_operators = [
        0.5
        * (
            np.asarray(operator, dtype=complex)
            + np.asarray(operator, dtype=complex).conj().T
        )
        for operator in operators
    ]
    if any(operator.shape != rho.shape for operator in hermitian_operators):
        raise ValueError("every operator must have the same shape as rho")
    count = len(hermitian_operators)
    if count == 0:
        return (
            0.0,
            np.empty(0),
            np.zeros_like(rho),
            np.empty((0, 0)),
            np.empty(0),
        )

    rho = 0.5 * (rho + rho.conj().T)
    drho = 0.5 * (drho + drho.conj().T)
    metric = np.empty((count, count), dtype=float)
    derivative = np.empty(count, dtype=float)
    for i, operator_i in enumerate(hermitian_operators):
        derivative[i] = float(np.real(np.trace(operator_i @ drho)))
        for j in range(i, count):
            value = float(
                np.real(np.trace(rho @ operator_i @ hermitian_operators[j]))
            )
            metric[i, j] = value
            metric[j, i] = value
    metric = 0.5 * (metric + metric.T)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    scale = max(float(np.max(eigenvalues)), np.finfo(float).tiny)
    retained = eigenvalues > rtol * scale
    inverse = np.zeros_like(eigenvalues)
    inverse[retained] = 1.0 / eigenvalues[retained]
    coefficients = eigenvectors @ (inverse * (eigenvectors.T @ derivative))
    reconstructed = sum(
        (
            coefficient * operator
            for coefficient, operator in zip(coefficients, hermitian_operators)
        ),
        start=np.zeros_like(rho),
    )
    reconstructed = 0.5 * (reconstructed + reconstructed.conj().T)
    captured_qfi = max(float(np.dot(coefficients, derivative)), 0.0)
    return captured_qfi, coefficients, reconstructed, metric, derivative


def build_configured_operator_bases(
    operators: dict[str, np.ndarray],
) -> dict[str, list[np.ndarray]]:
    """Build the standard linear and second-moment readout bases."""
    identity = operators["I"]
    Jx = operators["Jx"]
    Jy = operators["Jy"]
    linear = [identity, Jx, Jy]
    return {
        "Linear": linear,
        "Second moments": linear
        + [Jx @ Jx, Jy @ Jy, Jx @ Jy + Jy @ Jx],
    }


# ---------------------------------------------------------------------------
# Scaling fits
# ---------------------------------------------------------------------------


def fit_power_law(N: np.ndarray, FQ: np.ndarray) -> float:
    """Fit ``FQ ~ N**p`` on the upper half of the supplied sizes."""
    sizes = np.asarray(N, dtype=float)
    values = np.asarray(FQ, dtype=float)
    if sizes.ndim != 1 or values.shape != sizes.shape:
        raise ValueError("N and FQ must be one-dimensional arrays of equal length")

    midpoint = len(sizes) // 2
    start = max(0, midpoint - 1)
    valid = (sizes[start:] > 0.0) & (values[start:] > 0.0)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    exponent, _ = np.polyfit(
        np.log(sizes[start:][valid]),
        np.log(values[start:][valid]),
        1,
    )
    return float(exponent)


__all__ = [
    "DEFAULT_PLOT_OUTPUT_DIRECTORY",
    "GOOGLE_DRIVE_GRAPHS_DIRECTORY",
    "PLOT_SCHEMA_VERSION",
    "PlotRecord",
    "SUPPORTED_AXIS_SCALES",
    "SUPPORTED_COMPLEX_MODES",
    "SimulationConfig",
    "build_bath_operators",
    "build_configured_operator_bases",
    "build_hamiltonian",
    "build_initial_state",
    "build_spin_operators",
    "capture_params",
    "central_spin_state",
    "coherent_bath_state",
    "compute_bath_qfi_trajectory",
    "compute_qcrb_matrices",
    "fisher_metric_decomposition",
    "fisher_metric_projection",
    "fit_power_law",
    "frobenius_orthonormal_span",
    "get_bath_density_matrices",
    "observable_moment_fisher",
    "observable_projective_fisher",
    "observable_projective_score",
    "optimal_sz2_bath_state",
    "plot",
    "qfi_from_rho_and_drho",
    "qfi_vectorized",
    "save_plot",
]
