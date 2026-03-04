# score.py
import argparse
import math
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED, Lipinski


# -----------------------------
# IO
# -----------------------------
def read_smiles_from_smi(path: Path) -> list[str]:
    smiles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            smiles.append(line.split()[0])
    return smiles


def read_smiles_from_csv(path: Path, smiles_col: str) -> list[str]:
    df = pd.read_csv(path)
    if smiles_col not in df.columns:
        raise ValueError(f"Template CSV missing column: {smiles_col}. Available: {list(df.columns)}")
    s = df[smiles_col].dropna().astype(str).tolist()
    return [x.strip() for x in s if x.strip()]


# -----------------------------
# Utils
# -----------------------------
def clip01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def range_penalty(x: float, lo: float, hi: float) -> float:
    """超出范围才惩罚，惩罚归一到 0~1"""
    if lo <= x <= hi:
        return 0.0
    if x < lo:
        return clip01((lo - x) / max(lo, 1e-6))
    return clip01((x - hi) / max(hi, 1e-6))


def gaussian_score(x: float, mu: float, sigma: float) -> float:
    """高斯相似度（0~1），sigma 过小会自动兜底"""
    sigma = max(float(sigma), 1e-6)
    return math.exp(-((x - mu) ** 2) / (2.0 * sigma * sigma))


# -----------------------------
# Property computation
# -----------------------------
def compute_props(mol: Chem.Mol) -> dict:
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotb = Lipinski.NumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    frac_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    qed = float(QED.qed(mol))
    n_count = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N")
    heavy = rdMolDescriptors.CalcNumHeavyAtoms(mol)

    return {
        "MW": float(mw),
        "LogP": float(logp),
        "TPSA": float(tpsa),
        "HBD": int(hbd),
        "HBA": int(hba),
        "RotB": int(rotb),
        "Rings": int(rings),
        "FracCSP3": float(frac_csp3),
        "QED": float(qed),
        "N_count": int(n_count),
        "HeavyAtoms": int(heavy),
    }


def canonicalize(smi: str):
    m = Chem.MolFromSmiles(smi)
    if not m:
        return None
    return Chem.MolToSmiles(m, canonical=True)


# -----------------------------
# Small scoring
# -----------------------------
def calc_small_score(props: dict, cfg: dict) -> tuple[float, float, int]:
    """
    returns: (Score_small, Penalty, Violations)
    """
    # penalty uses "comfort ranges"
    pen = 0.0
    pen += range_penalty(props["MW"], cfg["pen_mw_min"], cfg["pen_mw_max"])
    pen += range_penalty(props["LogP"], cfg["pen_logp_min"], cfg["pen_logp_max"])
    pen += range_penalty(props["TPSA"], cfg["pen_tpsa_min"], cfg["pen_tpsa_max"])
    pen += range_penalty(props["RotB"], cfg["pen_rotb_min"], cfg["pen_rotb_max"])
    pen = pen / 4.0

    # violations (Lipinski/Veber style)
    vio = 0
    if props["MW"] > cfg["lip_mw_max"]:
        vio += 1
    if props["LogP"] > cfg["lip_logp_max"]:
        vio += 1
    if props["HBD"] > cfg["lip_hbd_max"]:
        vio += 1
    if props["HBA"] > cfg["lip_hba_max"]:
        vio += 1
    if props["TPSA"] > cfg["veb_tpsa_max"]:
        vio += 1
    if props["RotB"] > cfg["veb_rotb_max"]:
        vio += 1

    vmax = 6  # above 6 rules
    vio_norm = vio / vmax

    score = (
        cfg["w_qed"] * props["QED"]
        + cfg["w_fsp3"] * props["FracCSP3"]
        - cfg["w_pen"] * pen
        - cfg["w_vio"] * vio_norm
    )
    return float(score), float(pen), int(vio)


# -----------------------------
# Template scoring
# -----------------------------
def calc_template_stats(df: pd.DataFrame, features: list[str]) -> dict:
    stats = {}
    for f in features:
        if f not in df.columns:
            continue
        mu = float(df[f].mean())
        sigma = float(df[f].std(ddof=0))  # population std
        stats[f] = {"mu": mu, "sigma": max(sigma, 1e-6)}
    return stats


def calc_template_score(props: dict, stats: dict, weights: dict, raw_qed_w: float) -> tuple[float, dict]:
    """
    returns: (Score_tpl, per_feature_scores)
    """
    total = 0.0
    wsum = 0.0
    per = {}

    for f, w in weights.items():
        if f not in stats:
            continue
        x = float(props[f])
        mu = stats[f]["mu"]
        sigma = stats[f]["sigma"]
        s = gaussian_score(x, mu, sigma)
        per[f] = s
        total += float(w) * s
        wsum += float(w)

    # add raw QED
    total += float(raw_qed_w) * float(props["QED"])
    wsum += float(raw_qed_w)

    if wsum <= 0:
        return 0.0, per

    return float(total / wsum), per


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", choices=["property", "template"], required=True)

    ap.add_argument("-i", "--input", required=True, help="input .smi (one smiles per line)")
    ap.add_argument("-o", "--output_csv", required=True, help="output csv path")
    ap.add_argument("--output_smi", default="", help="optional output passed .smi")
    ap.add_argument("--dedup", action="store_true")

    # hard filters (applies to both modes)
    ap.add_argument("--mw_min", type=float, default=150)
    ap.add_argument("--mw_max", type=float, default=650)
    ap.add_argument("--logp_min", type=float, default=-1)
    ap.add_argument("--logp_max", type=float, default=5)
    ap.add_argument("--tpsa_min", type=float, default=20)
    ap.add_argument("--tpsa_max", type=float, default=140)
    ap.add_argument("--hbd_max", type=int, default=5)
    ap.add_argument("--hba_max", type=int, default=10)
    ap.add_argument("--rotb_max", type=int, default=10)
    ap.add_argument("--qed_min", type=float, default=0.3)

    ap.add_argument("--sort_by", default="Score")
    ap.add_argument("--descending", action="store_true")

    # small mode params
    ap.add_argument("--w_qed", type=float, default=0.70)
    ap.add_argument("--w_fsp3", type=float, default=0.10)
    ap.add_argument("--w_pen", type=float, default=0.15)
    ap.add_argument("--w_vio", type=float, default=0.05)

    ap.add_argument("--pen_mw_min", type=float, default=200)
    ap.add_argument("--pen_mw_max", type=float, default=600)
    ap.add_argument("--pen_logp_min", type=float, default=0)
    ap.add_argument("--pen_logp_max", type=float, default=4)
    ap.add_argument("--pen_tpsa_min", type=float, default=25)
    ap.add_argument("--pen_tpsa_max", type=float, default=120)
    ap.add_argument("--pen_rotb_min", type=float, default=0)
    ap.add_argument("--pen_rotb_max", type=float, default=8)

    ap.add_argument("--lip_mw_max", type=float, default=500)
    ap.add_argument("--lip_logp_max", type=float, default=5)
    ap.add_argument("--lip_hbd_max", type=int, default=5)
    ap.add_argument("--lip_hba_max", type=int, default=10)
    ap.add_argument("--veb_tpsa_max", type=float, default=140)
    ap.add_argument("--veb_rotb_max", type=int, default=10)

    # template mode params
    ap.add_argument("--template_csv", default="")
    ap.add_argument("--template_smiles_col", default="smiles")
    ap.add_argument("--tpl_features", default="MW,LogP,TPSA,N_count")
    ap.add_argument("--tpl_weights", default="MW:0.8,LogP:1.0,TPSA:0.8,N_count:0.6")
    ap.add_argument("--tpl_raw_qed_w", type=float, default=3.0)

    args = ap.parse_args()

    in_path = Path(args.input)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    smiles_raw = read_smiles_from_smi(in_path)

    # dedup
    smiles = []
    if args.dedup:
        seen = set()
        for s in smiles_raw:
            if "." in s:
                continue
            c = canonicalize(s)
            if not c:
                continue
            if c in seen:
                continue
            seen.add(c)
            smiles.append(c)
    else:
        smiles = [s for s in smiles_raw]

    # if template mode: prepare template stats
    tpl_stats = None
    tpl_weights = None
    tpl_features = None
    if args.mode == "template":
        if not args.template_csv:
            raise SystemExit("template mode requires --template_csv")
        tpl_smiles = read_smiles_from_csv(Path(args.template_csv), args.template_smiles_col)

        tpl_rows = []
        for s in tpl_smiles:
            if "." in s:
                continue
            m = Chem.MolFromSmiles(s)
            if not m:
                continue
            tpl_rows.append({"SMILES": s, **compute_props(m)})
        tpl_df = pd.DataFrame(tpl_rows)

        tpl_features = [x.strip() for x in args.tpl_features.split(",") if x.strip()]
        tpl_stats = calc_template_stats(tpl_df, tpl_features)

        # parse weights "MW:0.8,LogP:1.0"
        tpl_weights = {}
        for part in args.tpl_weights.split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            k, v = part.split(":", 1)
            tpl_weights[k.strip()] = float(v.strip())

    # scoring loop
    rows = []
    for s in smiles:
        if "." in s:
            continue
        m = Chem.MolFromSmiles(s)
        if not m:
            continue

        props = compute_props(m)

        # hard filters
        if not (args.mw_min <= props["MW"] <= args.mw_max):
            continue
        if not (args.logp_min <= props["LogP"] <= args.logp_max):
            continue
        if not (args.tpsa_min <= props["TPSA"] <= args.tpsa_max):
            continue
        if props["HBD"] > args.hbd_max:
            continue
        if props["HBA"] > args.hba_max:
            continue
        if props["RotB"] > args.rotb_max:
            continue
        if props["QED"] < args.qed_min:
            continue

        if args.mode == "property":
            cfg = {
                "w_qed": args.w_qed,
                "w_fsp3": args.w_fsp3,
                "w_pen": args.w_pen,
                "w_vio": args.w_vio,
                "pen_mw_min": args.pen_mw_min,
                "pen_mw_max": args.pen_mw_max,
                "pen_logp_min": args.pen_logp_min,
                "pen_logp_max": args.pen_logp_max,
                "pen_tpsa_min": args.pen_tpsa_min,
                "pen_tpsa_max": args.pen_tpsa_max,
                "pen_rotb_min": args.pen_rotb_min,
                "pen_rotb_max": args.pen_rotb_max,
                "lip_mw_max": args.lip_mw_max,
                "lip_logp_max": args.lip_logp_max,
                "lip_hbd_max": args.lip_hbd_max,
                "lip_hba_max": args.lip_hba_max,
                "veb_tpsa_max": args.veb_tpsa_max,
                "veb_rotb_max": args.veb_rotb_max,
            }
            score, pen, vio = calc_small_score(props, cfg)
            rows.append({
                "SMILES": Chem.MolToSmiles(m, isomericSmiles=True),
                **props,
                "Penalty": pen,
                "Violations": vio,
                "Score": score,
                "ScoreMode": "property",
            })
        else:
            score, per = calc_template_score(props, tpl_stats, tpl_weights, args.tpl_raw_qed_w)
            row = {
                "SMILES": Chem.MolToSmiles(m, isomericSmiles=True),
                **props,
                "Score": score,
                "ScoreMode": "template",
            }
            # optional: store per-feature similarity scores
            for k, v in per.items():
                row[f"Sim_{k}"] = float(v)
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["SMILES", "QED", "Score", "MW", "LogP", "TPSA", "HBD", "HBA", "RotB", "Rings", "FracCSP3", "N_count", "Penalty", "Violations", "ScoreMode"])

    # sort
    sort_by = args.sort_by if args.sort_by in df.columns else "Score"
    df = df.sort_values(by=sort_by, ascending=not args.descending).reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    if args.output_smi:
        out_smi = Path(args.output_smi)
        out_smi.parent.mkdir(parents=True, exist_ok=True)
        with open(out_smi, "w", encoding="utf-8") as f:
            for smi in df["SMILES"].tolist():
                f.write(smi + "\n")

    print(f"DONE mode={args.mode} input={len(smiles_raw)} used={len(smiles)} passed={len(df)} saved={out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

