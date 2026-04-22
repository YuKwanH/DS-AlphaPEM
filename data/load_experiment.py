import re
import pandas as pd


def load_aux_data_exp(project_root, stat_log_all, dyn_log_all, I_points):
    """Load and aggregate experimental REC sheet data into aux_data_exp.

    Returns a dict keyed by condition string (same keys as stat_log_all /
    dyn_log_all) with structure: {'states': {col_name: [value per I_point]}}.
    """
    raw_dir = project_root / "data" / "rawdata"
    sim_keys = sorted(set(stat_log_all) | set(dyn_log_all))
    p_to_tag = {1.3: 300, 1.4: 400, 1.5: 500}

    # Parse simulation conditions
    sim_cond = {}
    for key in sim_keys:
        m = re.search(r"RHC([0-9.]+)_P([0-9.]+)_T([0-9]+)", key)
        if m:
            sim_cond[key] = {
                "rhc": float(m.group(1)),
                "p_tag": p_to_tag.get(float(m.group(2))),
                "t": int(m.group(3)),
            }

    # Read REC sheets and summarise each sheet into one operating point
    rec_points_by_temp = {}
    for t in sorted({v["t"] for v in sim_cond.values()}):
        fpath = raw_dir / f"SYNTH_T{t}_N1.xlsx"
        if not fpath.exists():
            rec_points_by_temp[t] = []
            continue

        xls = pd.ExcelFile(fpath)
        rec_sheets = [s for s in xls.sheet_names if s.upper().startswith("REC_")]
        points = []

        for sname in rec_sheets:
            df = pd.read_excel(fpath, sheet_name=sname)
            num = df.apply(pd.to_numeric, errors="coerce")

            needed = [c for c in ["I_LOAD", "P_AIR", "HR_AIR_FC"] if c in num.columns]
            if len(needed) < 3:
                continue

            i_mean = float(num["I_LOAD"].mean())
            p_mean = float(num["P_AIR"].mean())
            hr_mean = float(num["HR_AIR_FC"].mean())

            p_tag = min([300, 400, 500], key=lambda p: abs(p_mean - p))
            rhc_tag = min([0.0, 0.5], key=lambda r: abs(hr_mean - 100 * r))

            col_means = {col: float(num[col].mean()) for col in num.columns if num[col].notna().any()}

            points.append({
                "sheet": sname,
                "i_mean": i_mean,
                "p_tag": p_tag,
                "rhc_tag": rhc_tag,
                "col_means": col_means,
            })

        rec_points_by_temp[t] = points

    # Build aux_data_exp in simulation-like format
    aux_data_exp = {}
    for key, cond in sim_cond.items():
        pts = [
            p for p in rec_points_by_temp.get(cond["t"], [])
            if p["p_tag"] == cond["p_tag"] and p["rhc_tag"] == cond["rhc"]
        ]

        if not pts:
            aux_data_exp[key] = {"states": {}}
            continue

        state_names = sorted(set().union(*(p["col_means"].keys() for p in pts)))
        states = {name: [] for name in state_names}

        for I in I_points:
            near = [p for p in pts if abs(p["i_mean"] - I) <= 2.0]
            if not near:
                near = [min(pts, key=lambda p: abs(p["i_mean"] - I))]

            for name in state_names:
                vals = [p["col_means"].get(name) for p in near]
                vals = [v for v in vals if v is not None and pd.notna(v)]
                states[name].append(float(sum(vals) / len(vals)) if vals else float("nan"))

        aux_data_exp[key] = {"states": states}

    return aux_data_exp
