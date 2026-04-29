"""Save & download box.

Sits below the result section. Lets the user pick a format (CSV / NumPy /
Excel), specify a directory and filename, then either write the file to
disk on the local machine or trigger a browser download.
"""

import io
import os
import time

import numpy as np
import pandas as pd
import streamlit as st


FORMATS = {
    "CSV": (".csv", "text/csv"),
    "NumPy (.npz)": (".npz", "application/octet-stream"),
    "Excel (.xlsx)": (
        ".xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
}


def render(state):
    st.markdown("##### 💾 Save & download result")

    res = state.get("last_result")
    if not res or not res.get("status", {}).get("success"):
        st.caption("Run a simulation first — then come back here to save the result.")
        return

    cols = st.columns([1, 2])
    fmt = cols[0].selectbox(
        "Format",
        list(FORMATS.keys()),
        index=list(FORMATS.keys()).index(state.get("save_fmt", "CSV")),
        key="save_fmt_select",
    )
    state["save_fmt"] = fmt
    ext, mime = FORMATS[fmt]

    default_dir = state.get("save_dir") or os.getcwd()
    save_dir = cols[1].text_input(
        "Directory",
        value=default_dir,
        key="save_dir_input",
        help="Local path where the file will be written when you click Save.",
    )
    state["save_dir"] = save_dir

    default_name = _make_filename(state, ext)
    save_name = st.text_input(
        "Filename",
        value=default_name,
        key=f"save_name_input_{ext}",
    )
    if not save_name.endswith(ext):
        save_name += ext

    try:
        payload = _serialize(res, fmt)
    except Exception as exc:
        st.error(f"Could not prepare data for {fmt}: {exc}")
        return

    bcols = st.columns(2)
    if bcols[0].button("💾 Save to disk", key="save_disk_button", use_container_width=True):
        try:
            os.makedirs(save_dir, exist_ok=True)
            full_path = os.path.join(save_dir, save_name)
            with open(full_path, "wb") as f:
                f.write(payload)
            st.success(f"Saved {len(payload) / 1024:.1f} kB → `{full_path}`")
        except Exception as exc:
            st.error(f"Save failed: {exc}")

    bcols[1].download_button(
        "⬇ Download",
        data=payload,
        file_name=save_name,
        mime=mime,
        key="save_download_button",
        use_container_width=True,
    )


def _make_filename(state, ext):
    variant = state.get("model_variant", "run").lower().replace(" ", "_").replace("-", "")
    profile = state.get("profile_kind", "x").lower().replace(" ", "_")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"pemfc_{variant}_{profile}_{stamp}{ext}"


def _serialize(res, fmt):
    status = res["status"]
    ext, _ = FORMATS[fmt]

    if status["kind"] == "polar":
        polar = res["polar"]
        df = pd.DataFrame({
            "i_A_per_m2": polar["i_A_m2"],
            "i_A_per_cm2": polar["i_A_m2"] / 1e4,
            "Ucell_V": polar["Ucell_V"],
        })
        if fmt == "CSV":
            return df.to_csv(index=False).encode("utf-8")
        if fmt == "NumPy (.npz)":
            buf = io.BytesIO()
            np.savez_compressed(buf, **{c: df[c].to_numpy() for c in df.columns})
            return buf.getvalue()
        if fmt == "Excel (.xlsx)":
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                df.to_excel(w, sheet_name="polarization", index=False)
            return buf.getvalue()

    model = res["model"]
    variables = _to_dataframe(getattr(model, "variables", {}))
    fluxes = _to_dataframe(getattr(model, "fluxes", {}))
    echem = _to_dataframe(getattr(model, "echem_traj", {}))

    if fmt == "CSV":
        merged = pd.concat(
            [
                variables.add_prefix("var_"),
                fluxes.add_prefix("flux_"),
                echem.add_prefix("echem_"),
            ],
            axis=1,
        )
        return merged.to_csv(index=False).encode("utf-8")

    if fmt == "NumPy (.npz)":
        buf = io.BytesIO()
        bundle = {}
        for prefix, src in (("var_", getattr(model, "variables", {})),
                            ("flux_", getattr(model, "fluxes", {})),
                            ("echem_", getattr(model, "echem_traj", {}))):
            for k, v in src.items():
                arr = np.asarray(v) if hasattr(v, "__len__") else np.array([v])
                bundle[f"{prefix}{k}"] = arr
        np.savez_compressed(buf, **bundle)
        return buf.getvalue()

    if fmt == "Excel (.xlsx)":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            if not variables.empty:
                variables.to_excel(w, sheet_name="variables", index=False)
            if not fluxes.empty:
                fluxes.to_excel(w, sheet_name="fluxes", index=False)
            if not echem.empty:
                echem.to_excel(w, sheet_name="echem_traj", index=False)
        return buf.getvalue()

    raise ValueError(f"Unknown format: {fmt}")


def _to_dataframe(d):
    """Build a DataFrame from a dict of time-series.

    Skips non-1D entries (some fluxes are stored as nested lists). For 1D
    entries shorter than the longest column, pads with NaN; for entries
    longer than the longest column, truncates.
    """
    if not d:
        return pd.DataFrame()
    flattened = {}
    for k, v in d.items():
        if not hasattr(v, "__len__"):
            flattened[k] = np.array([v], dtype=float)
            continue
        try:
            arr = np.asarray(v)
        except Exception:
            continue
        if arr.dtype == object:
            # Ragged array — try to coerce each element to scalar; skip if not scalar.
            try:
                arr = np.array([float(x) if np.isscalar(x) else np.nan for x in v], dtype=float)
            except Exception:
                continue
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim == 1:
            flattened[k] = arr
        else:
            # 2-D entry, e.g. per-node profile recorded over time. Expand
            # the second axis into separate columns.
            for i in range(arr.shape[1] if arr.ndim == 2 else 0):
                flattened[f"{k}[{i}]"] = arr[:, i]
    if not flattened:
        return pd.DataFrame()
    n = max(len(v) for v in flattened.values())
    cols = {}
    for k, v in flattened.items():
        if len(v) == n:
            cols[k] = v
        elif len(v) == 0:
            cols[k] = np.full(n, np.nan)
        elif len(v) < n:
            padded = np.full(n, np.nan, dtype=float)
            padded[: len(v)] = v
            cols[k] = padded
        else:
            cols[k] = v[:n]
    return pd.DataFrame(cols)
