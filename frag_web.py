import os
import csv
import re
import subprocess
from pathlib import Path
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import Draw
from streamlit_option_menu import option_menu
from datetime import datetime
# ==============================
# 基本配置
# ==============================
PROJECT_ROOT = Path(__file__).parent.resolve()
TEST_SCRIPT = PROJECT_ROOT / "frag.py"
MACFRAG_SCRIPT = PROJECT_ROOT / "MacFrag.py"
WORKDIR = PROJECT_ROOT / "workspace"
WORKDIR.mkdir(exist_ok=True)
LINK_SCRIPT = PROJECT_ROOT / "link.py"
SCORE_SCRIPT = PROJECT_ROOT / "score.py"
# 临时上传目录（中转用）
TEMP_INPUT_DIR = WORKDIR / "temp_input"
TEMP_INPUT_DIR.mkdir(exist_ok=True)
# ==============================
# 工具函数
# ==============================
def save_upload(file, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    path = dst_dir / file.name
    with open(path, "wb") as f:
        f.write(file.getbuffer())
    return path


def run_cmd(cmd):
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = p.communicate()
    return p.returncode, out, err

def run_cmd_stream(cmd, on_line=None):
    """
    逐行读取 stdout，用于进度可视化
    """
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in p.stdout:
        line = line.rstrip("\n")
        if on_line:
            on_line(line)
    p.wait()
    return p.returncode


def read_original_molecule(path: Path):
    if path.suffix == ".smi":
        with open(path) as f:
            smi = f.readline().split()[0]
        return Chem.MolFromSmiles(smi)

    if path.suffix == ".sdf":
        suppl = Chem.SDMolSupplier(str(path), removeHs=False)
        return suppl[0] if suppl and suppl[0] else None
    return None


def read_fragments_smi(path: Path):
    mols = []
    with open(path) as f:
        for line in f:
            smi = line.strip().split()[0]
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mols.append(mol)
    return mols

def read_smiles_from_smi(path: Path):
    smiles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            smi = line.split()[0]
            smiles.append(smi)
    return smiles


def read_smiles_from_csv(path: Path, smiles_col: str = "smiles"):
    smiles = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if smiles_col not in reader.fieldnames:
            raise ValueError(f"CSV 中找不到列: {smiles_col}, 当前列={reader.fieldnames}")
        for row in reader:
            smi = (row.get(smiles_col) or "").strip()
            if smi:
                smiles.append(smi)
    return smiles

def collect_frag_files_from_jobs(job_info: dict, methods: list[str]):
    """
    从 st.session_state["job_info"] 里收集碎片 .smi 文件
    methods: ["Brics","Recap","Macfrag"]
    """
    files = []
    if not job_info:
        return files

    for mol_name, info in job_info.items():
        out_dir = Path(info["output_dir"])
        for m in methods:
            pattern = f"*{m.lower()}*.smi"
            files.extend(out_dir.rglob(pattern))
    return sorted(set(files))

_dummy_pat = re.compile(r"\[\d+\*\]")  # 匹配 [1*] [14*] [123*]...

def clean_fragment_smiles(smi: str) -> str:
    """将 [n*] 统一替换为 [*]，用于建库时统一dummy标记"""
    if not smi:
        return smi
    return _dummy_pat.sub("[*]", smi)

def normalize_and_canonicalize(smi: str) -> str | None:
    """清洗dummy后，转mol并输出canonical smiles；失败则返回None"""
    smi2 = clean_fragment_smiles(smi.strip())
    mol = Chem.MolFromSmiles(smi2)
    if mol is None:
        return None
    # canonical 的输出会稳定，利于去重
    return Chem.MolToSmiles(mol, canonical=True)

def write_fragment_library_csv(smiles_list, out_csv: Path, meta: dict | None = None):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat(timespec="seconds")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["smiles", "created_at", "lib_name"])
        writer.writeheader()
        for smi in smiles_list:
            writer.writerow({
                "smiles": smi,
                "created_at": now,
                "lib_name": (meta or {}).get("lib_name", "default"),
            })

def write_fragment_library_smi(smiles_list, out_smi: Path):
    out_smi.parent.mkdir(parents=True, exist_ok=True)
    with open(out_smi, "w", encoding="utf-8") as f:
        for smi in smiles_list:
            f.write(smi + "\n")

def weights_dict_to_str(d: dict) -> str:
    # {"MW":0.8,"LogP":1.0} -> "MW:0.8,LogP:1.0"
    parts = []
    for k, v in d.items():
        try:
            vv = float(v)
        except Exception:
            continue
        parts.append(f"{k}:{vv}")
    return ",".join(parts)

# ==============================
# 单分子 → SVG（统一风格）
# ==============================
def mol_to_svg(mol, size=(300, 300)):
    drawer = Draw.MolDraw2DSVG(size[0], size[1])
    opts = drawer.drawOptions()
    opts.clearBackground = False
    # opts.useBWAtomPalette() 开启后分子显示黑白
    opts.padding = 0.01
    # 统一键风格
    opts.fixedBondLength = 100
    opts.bondLineWidth = 2.0
    opts.scaleBondWidth = False
    opts.addAtomIndices = False
    opts.addBondIndices = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return "\n".join(svg.splitlines()[1:])
# ==============================
# 多分子 → SVG Grid（HTML）
# ==============================
def fragments_to_svg_grid_html(
    mols,
    mols_per_row=4,
    sub_size=(260, 260),
):
    if not mols:
        return ""

    cards = []
    for mol in mols:
        svg = mol_to_svg(mol, size=sub_size)
        cards.append(
            f"""
            <div style="
                border:1px solid #e0e0e0;
                border-radius:12px;
                padding:10px;
                background:#ffffff;
                box-shadow:0 2px 6px rgba(0,0,0,0.06);
                display:flex;
                align-items:center;
                justify-content:center;
            ">
                {svg}
            </div>
            """
        )

    return f"""
    <div style="
        display:grid;
        grid-template-columns: repeat({mols_per_row}, 1fr);
        gap:16px;
    ">
        {''.join(cards)}
    </div>
    """
# ==============================
# Streamlit UI (Clean Layout)
# ==============================
st.set_page_config(page_title="Molecular Workflow", layout="wide")
st.title("🧩 Molecular workflow")
with st.sidebar:
    st.header("📌 Navigation")
    
    
    selected_module = option_menu(
        menu_title=None,
        options=["分子碎片化", "分子碎片库", "碎片重组", "打分筛选"],
        icons=["bezier2", "collection", "shuffle", "star"],
        default_index=0,
        key="module_selector",
        styles={
            "container": {"padding": "6px 0px"},
            "icon": {"font-size": "18px", "margin-right": "10px"},
            "nav-link": {
                "font-size": "16px",
                "padding": "10px 14px",
                "margin": "6px 0px",
                "border-radius": "10px",
            },
            "nav-link-selected": {"background-color": "#ff002f"},
        },
    )
    score_mode = None
    if selected_module == "打分筛选":
        # 小号标题（科研风）
        st.markdown(
            "<div style='text-align:center; font-size:18px; font-weight:600; margin:8px 0 6px 0; color:#555;'>Scoring Mode</div>",
            unsafe_allow_html=True,
        )

        score_mode = option_menu(
            menu_title=None,   # ⬅️ 不用内置 title
            options=[
                "Property-Based",
                "Template-Match",
            ],
            icons=["speedometer2", "diagram-3"],
            default_index=0,
            key="score_mode_nav",
            styles={
                "container": {"padding": "0px"},
                "icon": {"font-size": "14px", "margin-right": "6px"},
                "nav-link": {
                    "font-size": "14px",
                    "padding": "6px 8px",
                    "border-radius": "6px",
                },
                "nav-link-selected": {"background-color": "#ff002f"},
            },
        )

# ==============================
# Helper: nice section container
# ==============================
def card(title: str, desc: str = ""):
    with st.container(border=True):
        st.markdown(f"### {title}")
        if desc:
            st.caption(desc)
        return st

# ==============================
# Module 1: 分子碎片化（参数在主页面左列）
# ==============================
def render_fragmentation():
    st.markdown("## 1) 分子碎片化")
    st.caption("上传分子文件，选择碎片化方法（Brics / Recap / Macfrag），生成碎片并可视化。")

    # =========================
    # 上：参数设置 + 启动
    # =========================
    with st.container(border=True):
        st.markdown("### ⚙️ 参数设置")
        st.caption("选择方法与 MacFrag 参数，上传输入文件，然后点击 Start。")

        c1, c2, c3 = st.columns([1.2, 1.2, 0.8], gap="large")

        with c1:
            methods = st.multiselect(
                "碎片化方法",
                ["Brics", "Recap", "Macfrag"],
                default=["Brics"],
                key="frag_methods",
            )

            if "Macfrag" in methods:
                st.markdown("**MacFrag 参数**")
                maxBlocks = st.number_input("maxBlocks", 1, 20, 6, key="mac_maxBlocks")
                maxSR = st.number_input("maxSR", 3, 20, 8, key="mac_maxSR")
                minFragAtoms = st.number_input("minFragAtoms", 1, 20, 1, key="mac_minFragAtoms")
                asMols = st.selectbox("asMols", ["False", "True"], key="mac_asMols")
            else:
                maxBlocks, maxSR, minFragAtoms, asMols = 6, 8, 1, "False"

        with c2:
            st.markdown("**📤 上传文件**")
            smi_files = st.file_uploader("Input .smi", type=["smi"], accept_multiple_files=True, key="frag_smi")
            sdf_files = st.file_uploader("Input .sdf", type=["sdf"], accept_multiple_files=True, key="frag_sdf")

        with c3:
            st.markdown("**🚀 运行**")
            run_btn = st.button("Start", width="stretch", key="frag_run")
            st.caption("点击后开始碎片化并生成结果。")

    # =========================
    # 下：结果展示 / 预览
    # =========================
    with st.container(border=True):
        st.markdown("### 🧪 运行结果")
        st.caption("生成的碎片与原分子预览会显示在这里。")

        # 运行碎片化
        if run_btn:
            uploaded = []
            for f in smi_files or []:
                uploaded.append(save_upload(f, TEMP_INPUT_DIR))
            for f in sdf_files or []:
                uploaded.append(save_upload(f, TEMP_INPUT_DIR))

            if not uploaded:
                st.error("未上传分子")
                st.stop()

            st.session_state["job_info"] = {}
            pid = os.getpid()

            for fp in uploaded:
                file_basename = fp.stem
                job_dir = WORKDIR / f"job_{file_basename}_{pid}"
                inp_dir = job_dir / "input"
                out_dir = job_dir / "output"
                inp_dir.mkdir(parents=True, exist_ok=True)
                out_dir.mkdir(parents=True, exist_ok=True)

                final_input_path = inp_dir / fp.name
                final_input_path.write_bytes(fp.read_bytes())

                for m in methods:
                    run_cmd([
                        "python",
                        str(TEST_SCRIPT),
                        "-i", str(final_input_path),
                        "-o", str(out_dir),
                        "-m", m,
                    ])

                if "Macfrag" in methods:
                    mac_out = out_dir / "macfrag"
                    mac_out.mkdir(exist_ok=True)
                    run_cmd([
                        "python", str(MACFRAG_SCRIPT),
                        "-i", str(final_input_path),
                        "-o", str(mac_out),
                        "-maxBlocks", str(maxBlocks),
                        "-maxSR", str(maxSR),
                        "-minFragAtoms", str(minFragAtoms),
                        "-asMols", asMols,
                    ])
                    redundant = mac_out / "macfrag_frag.smi"
                    if redundant.exists():
                        redundant.unlink()

                st.session_state["job_info"][file_basename] = {
                    "job_dir": str(job_dir),
                    "input_file": str(final_input_path),
                    "output_dir": str(out_dir),
                }

            st.success("🎉 Complete!")

        # 可视化
        if "job_info" in st.session_state and st.session_state["job_info"]:
            st.markdown("#### ② Visualization")
            molecule_names = list(st.session_state["job_info"].keys())
            sel_mol_name = st.selectbox("选择分子", molecule_names, key="frag_sel_mol")

            job_info = st.session_state["job_info"][sel_mol_name]
            input_file = Path(job_info["input_file"])
            output_dir = Path(job_info["output_dir"])

            mol = read_original_molecule(input_file)
            if not mol:
                st.error(f"无法解析分子文件：{input_file.name}")
                st.stop()

            tabs = st.tabs(["Brics", "Recap", "Macfrag"])
            for tab, method in zip(tabs, ["Brics", "Recap", "Macfrag"]):
                with tab:
                    frag_files = list(output_dir.rglob(f"*{method.lower()}*.smi"))
                    if not frag_files:
                        st.info(f"{method} 未为 {sel_mol_name} 生成碎片")
                        continue

                    frags = read_fragments_smi(frag_files[0])
                    if not frags:
                        st.warning(f"{method} 为 {sel_mol_name} 生成的碎片为空")
                        continue

                    st.markdown(f"**🧬 Original molecule：{input_file.name}**")
                    components.html(
                        fragments_to_svg_grid_html([mol], mols_per_row=3, sub_size=(400, 300)),
                        height=350,
                    )

                    st.markdown("---")

                    st.markdown(f"**🧩 {method} fragments（{len(frags)}）**")
                    components.html(
                        fragments_to_svg_grid_html(frags, mols_per_row=5, sub_size=(260, 260)),
                        height=900,
                        scrolling=True,
                    )
        else:
            st.info("还没有运行任务。请在上方参数区上传分子并点击 Start。")

# ==============================
# Module 2: 分子碎片库（新页面骨架，参数在左列）
# ==============================
def render_fragment_library():
    st.markdown("## 2) 分子碎片库")
    st.caption("从上阶段碎片结果合并，或上传碎片文件/CSV，构建碎片库并预览结构。")

    # =========================
    # 上：参数设置 + 构建
    # =========================
    with st.container(border=True):
        st.markdown("### 📥 导入与构建")
        st.caption("选择导入来源，配置去重与库名称，然后点击构建。")

        c1, c2, c3 = st.columns([1.2, 1.2, 0.8], gap="large")

        with c1:
            import_mode = st.radio(
                "导入来源",
                ["从上阶段碎片结果合并", "上传碎片文件（.smi / CSV）"],
                index=0,
                key="lib_import_mode",
            )
            dedup = st.checkbox("SMILES 去重", value=True, key="lib_dedup")
            lib_name = st.text_input("库名称", value="default", key="lib_name")

        with c2:
            selected_methods = []
            merge_files = []
            upload_smi_files = []
            upload_csv_files = []
            csv_smiles_col = "smiles"

            if import_mode == "从上阶段碎片结果合并":
                st.markdown("**🧩 合并上阶段碎片**")
                selected_methods = st.multiselect(
                    "碎片方法",
                    ["Brics", "Recap", "Macfrag"],
                    default=["Brics"],
                    key="lib_merge_methods",
                )

                job_info = st.session_state.get("job_info", {})
                if not job_info:
                    st.warning("检测不到上阶段 job_info（请先在“分子碎片化”模块运行生成碎片）。")
                    available_files = []
                else:
                    available_files = collect_frag_files_from_jobs(job_info, selected_methods)

                file_labels = [str(p) for p in available_files]
                chosen_files = st.multiselect(
                    "选择要合并的碎片文件",
                    options=file_labels,
                    default=file_labels[: min(5, len(file_labels))],
                    key="lib_merge_files",
                )
                merge_files = [Path(x) for x in chosen_files]

            else:
                st.markdown("**📤 上传碎片文件**")
                upload_smi_files = st.file_uploader(
                    "上传碎片 .smi（可多选）",
                    type=["smi"],
                    accept_multiple_files=True,
                    key="lib_upload_smi",
                )
                upload_csv_files = st.file_uploader(
                    "上传 CSV（可多选）",
                    type=["csv"],
                    accept_multiple_files=True,
                    key="lib_upload_csv",
                )
                csv_smiles_col = st.text_input(
                    "CSV SMILES 列名",
                    value="smiles",
                    key="lib_csv_smiles_col",
                )

        with c3:
            st.markdown("**🔨 构建**")
            build_btn = st.button("构建/更新库", width="stretch", key="lib_build_btn")
            st.caption("构建后会在下方预览碎片结构。")

        st.session_state["fraglib_cfg"] = {
            "import_mode": import_mode,
            "dedup": dedup,
            "lib_name": lib_name,
            "selected_methods": selected_methods,
            "csv_smiles_col": csv_smiles_col,
        }

        # 构建逻辑
        if build_btn:
            raw_smiles = []

            # A) merge 上阶段碎片
            if import_mode == "从上阶段碎片结果合并":
                if not merge_files:
                    st.error("未选择任何上阶段碎片文件。")
                else:
                    for fp in merge_files:
                        try:
                            raw_smiles.extend(read_smiles_from_smi(fp))
                        except Exception as e:
                            st.warning(f"读取失败: {fp.name} ({e})")

            # B) 用户上传 smi/csv
            else:
                for uf in upload_smi_files or []:
                    tmp_path = save_upload(uf, WORKDIR / "fraglib_uploads" / "smi")
                    raw_smiles.extend(read_smiles_from_smi(tmp_path))

                for cf in upload_csv_files or []:
                    tmp_path = save_upload(cf, WORKDIR / "fraglib_uploads" / "csv")
                    try:
                        raw_smiles.extend(read_smiles_from_csv(tmp_path, smiles_col=csv_smiles_col))
                    except Exception as e:
                        st.error(str(e))

            # ===== 统一 dummy + canonical 去重 =====
            unique = {}
            bad = 0
            for smi in raw_smiles:
                canon = normalize_and_canonicalize(smi)
                if canon is None:
                    bad += 1
                    continue
                unique[canon] = canon

            final_smiles = list(unique.values())

            st.session_state["fraglib_smiles"] = final_smiles

            # ===== 写出库文件（CSV + SMI）=====
            lib_dir = WORKDIR / "fragment_library"
            csv_path = lib_dir / f"{lib_name}.csv"
            smi_path = lib_dir / f"{lib_name}.smi"

            write_fragment_library_csv(final_smiles, csv_path, meta={"lib_name": lib_name})
            write_fragment_library_smi(final_smiles, smi_path)

            st.success(f"碎片库构建完成：输入 {len(raw_smiles)} 条，解析失败 {bad} 条，去重后 {len(final_smiles)} 条")
            st.info(f"已保存：{csv_path} 以及 {smi_path}")

    # =========================
    # 下：预览区
    # =========================
    with st.container(border=True):
        st.markdown("### 🧾 碎片库预览")
        st.caption("默认展示前 20 个碎片，可调整数量。")

        smiles_all = st.session_state.get("fraglib_smiles", [])
        total_n = len(smiles_all)

        if total_n == 0:
            st.info("尚未构建碎片库。请在上方点击“构建/更新库”。")
            return

        # 展示数量（默认 20，但可增大）
        show_n = st.slider(
            "当前展示碎片数量",
            min_value=1,
            max_value=min(200, total_n),
            value=min(20, total_n),
            step=1,
            key="lib_show_n",
        )
        st.caption(f"当前库中共有 {total_n} 个碎片，正在展示前 {show_n} 个")

        preview_smiles = smiles_all[:show_n]

        mols = []
        bad = 0
        for smi in preview_smiles:
            m = Chem.MolFromSmiles(smi)
            if m:
                mols.append(m)
            else:
                bad += 1

        if bad:
            st.warning(f"有 {bad} 条 SMILES 无法解析，已跳过。")

        if mols:
            components.html(
                fragments_to_svg_grid_html(mols, mols_per_row=5, sub_size=(260, 260)),
                height=900,
                scrolling=True,
            )
        else:
            st.warning("没有可展示的有效碎片。")
        
        smi_text = "\n".join(smiles_all) + "\n"
        st.download_button(
            label="⬇️ 下载碎片库（.smi）",
            data=smi_text,
            file_name=f"{st.session_state.get('lib_name', 'fragment_library')}.smi",
            mime="text/plain",
        )

# ==============================
# Module 3/4: 先占位（同样布局）
# ==============================
def render_reassembly():
    st.markdown("## 3) 碎片重组")
    st.caption("默认使用模块2构建的碎片库（或上传 .smi 碎片库），两两拼接生成新分子，并实时显示进度。")
    # =========================
    # 上：参数 + 启动
    # =========================
    with st.container(border=True):
        st.markdown("### ⚙️ 重组参数")
        c1, c2, c3 = st.columns([1.2, 1.2, 0.9], gap="large")

        with c1:
            source_mode = st.radio(
                "碎片来源",
                ["使用已构建碎片库（模块2）", "上传碎片库（.smi）"],
                index=0,
                key="re_source_mode",
            )
            upload_lib = None
            if source_mode == "上传碎片库（.smi）":
                upload_lib = st.file_uploader(
                    "上传碎片库 .smi（每行一个 SMILES）",
                    type=["smi"],
                    key="re_upload_smi",
                )
            # ✅ 新增：规则模式选择
            re_mode = st.selectbox(
                "连接规则模式",
                options=["strict", "loose"],
                index=0,
                key="re_rule_mode",
            )
            st.caption("strict：优先更合理的连接；loose：放宽规则以获得更多结构多样性。")
        
        with c2: 
            max_frags = st.slider("最多使用碎片数", 20, 2000, 300, step=10, key="re_max_frags")
            max_pairs = st.slider("最多尝试 pairs 数", 100, 200000, 10000, step=100, key="re_max_pairs")
            topk = st.slider("最多保留结果数（TopK）", 50, 5000, 1000, step=50, key="re_topk")
            seed = st.number_input("随机种子（用于采样 pairs）", 0, 999999, 0, step=1, key="re_seed")

        with c3:
            st.markdown("**🚀 运行**")
            run_btn = st.button("开始重组", width="stretch", key="re_run_btn")
            st.caption("运行时会显示进度，结束后可预览并下载。")

    # =========================
    # 下：结果展示 / 预览 / 下载
    # =========================
    with st.container(border=True):
        st.markdown("### 🧪 重组结果")
        st.caption("实时日志与进度会显示在这里。")

        # 选择碎片库输入
        frag_smiles = []
        if source_mode == "使用已构建碎片库（模块2）":
            frag_smiles = st.session_state.get("fraglib_smiles", [])
            if not frag_smiles:
                st.info("未检测到模块2碎片库，请先到模块2构建碎片库，或选择上传 .smi。")
                return
        else:
            if upload_lib is None:
                st.info("请先上传碎片库 .smi 文件。")
                return
            # 直接读上传内容
            content = upload_lib.getvalue().decode("utf-8", errors="ignore")
            for line in content.splitlines():
                line = line.strip()
                if line:
                    frag_smiles.append(line.split()[0])

        # 准备 I/O 文件（写入 workspace）
        re_dir = WORKDIR / "reassembly"
        re_dir.mkdir(parents=True, exist_ok=True)

        lib_in = re_dir / "input_fraglib.smi"
        lib_out = re_dir / "recombined_top.smi"

        if run_btn:
            # ✅ 新增：把 UI 选项映射为 link.py 参数
            rule_mode = "strict" if re_mode.startswith("strict") else "loose"

            # 写入输入库文件
            with open(lib_in, "w", encoding="utf-8") as f:
                for s in frag_smiles:
                    f.write(s + "\n")

            prog = st.progress(0)
            status = st.empty()
            log_box = st.empty()

            def on_line(line: str):
                log_box.write(line)

                if line.startswith("PROGRESS "):
                    # PROGRESS k/total valid=xx
                    part = line.split()[1]  # k/total
                    k, total = part.split("/")
                    k = int(k)
                    total = int(total)
                    prog.progress(int(k / total * 100))
                elif line.startswith("INFO:"):
                    status.info(line)
                elif line.startswith("DONE"):
                    status.success(line)
                elif line.startswith("ERROR"):
                    status.error(line)

            rc = run_cmd_stream([
                "python",
                str(LINK_SCRIPT),
                "-i", str(lib_in),
                "-o", str(lib_out),
                "--max_frags", str(int(max_frags)),
                "--max_pairs", str(int(max_pairs)),
                "--topk", str(int(topk)),
                "--seed", str(int(seed)),
                "--mode", rule_mode,  # ✅ 新增：传给 link.py
            ], on_line=on_line)
            
            if rc != 0:
                st.error("重组失败，请查看日志输出。")
                return
      

            # 读取结果并缓存
            data = lib_out.read_text(encoding="utf-8")
            st.session_state["recombined_smi_text"] = data

            # 解析用于可视化
            mols = []
            scored = []
            for line in data.splitlines():
                parts = line.split("\t")
                smi = parts[0].strip()
                qed = float(parts[1]) if len(parts) > 1 else None
                m = Chem.MolFromSmiles(smi)
                if m:
                    mols.append(m)
                    scored.append((smi, qed))

            st.session_state["recombined_mols"] = mols
            st.session_state["recombined_scored"] = scored

            st.success("✅ 重组完成，已载入结果用于预览与下载。")

        # ===== 展示已有结果（不一定本次运行生成）=====
        mols = st.session_state.get("recombined_mols", [])
        smi_text = st.session_state.get("recombined_smi_text", "")

        if not mols:
            st.info("暂无结果。请点击上方“开始重组”。")
            return

        show_top = st.slider("预览 Top N", 10, min(200, len(mols)), min(50, len(mols)), step=10, key="re_show_top")
        mols_show = mols[:show_top]

        components.html(
            fragments_to_svg_grid_html(mols_show, mols_per_row=5, sub_size=(260, 260)),
            height=900,
            scrolling=True,
        )

        st.download_button(
            "⬇️ 下载重组结果（SMI + QED）",
            data=smi_text,
            file_name="recombined_top.smi",
            mime="text/plain",
        )


def render_scoring(score_mode: str):
    st.markdown("## 4) 打分筛选")
    st.caption("Property：按成药性/理化属性筛潜力；Template：按模板库统计分布相似度打分。")

    # =========================
    # 统一：输入来源
    # =========================
    with st.container(border=True):
        st.markdown("### 📥 输入")
        c1, c2 = st.columns([1.2, 1.0], gap="large")

        with c1:
            source_mode = st.radio(
                "输入来源",
                ["使用模块3重组结果", "上传待打分 .smi"],
                index=0,
                key="score_source_mode",
            )
            upload_smi = None
            if source_mode == "上传待打分 .smi":
                upload_smi = st.file_uploader(
                    "上传 .smi（每行一个 SMILES）",
                    type=["smi"],
                    key="score_upload_smi",
                )

        with c2:
            topk = st.slider("输出保留 TopK", 50, 20000, 2000, step=50, key="score_topk")
            sort_desc = st.checkbox("按分数降序（高分在前）", value=True, key="score_desc")
            preview_n = st.slider("结构预览 Top N", 10, 200, 50, 10, key="score_preview_n")

    # =========================
    # 模式隔离：Property / Template
    # =========================
    mode_key = "property" if score_mode.startswith("Property-Based") else "template"

    # 输出目录（按模式分开，避免互相覆盖）
    score_dir = WORKDIR / "scoring" / mode_key
    score_dir.mkdir(parents=True, exist_ok=True)

    in_smi = score_dir / "input_to_score.smi"
    out_csv = score_dir / "scored.csv"
    out_smi = score_dir / "scored_top.smi"

    # ========== 收集输入 SMILES ==========
    smiles_list = []
    if source_mode == "使用模块3重组结果":
        smi_text = st.session_state.get("recombined_smi_text", "")
        if smi_text.strip():
            for line in smi_text.splitlines():
                s = line.split("\t")[0].strip()
                if s:
                    smiles_list.append(s)
        else:
            default_file = WORKDIR / "reassembly" / "recombined_top.smi"
            if not default_file.exists():
                st.info("未检测到模块3输出，请先在模块3生成重组结果，或选择上传 .smi。")
                return
            for line in default_file.read_text(encoding="utf-8").splitlines():
                s = line.split("\t")[0].strip()
                if s:
                    smiles_list.append(s)
    else:
        if upload_smi is None:
            st.info("请先上传待打分 .smi。")
            return
        content = upload_smi.getvalue().decode("utf-8", errors="ignore")
        for line in content.splitlines():
            line = line.strip()
            if line:
                smiles_list.append(line.split()[0])

    # =========================
    # Small 模式 UI + 运行
    # =========================
    if mode_key == "property":
        with st.container(border=True):
            st.markdown("### 🧪 Property 打分（成药性/理化属性）")
            st.caption("建议用于小样本快速筛潜力：分数 = w_qed*QED + w_fsp3*FracCSP3 - w_pen*Penalty - w_vio*Violations_norm")

            c1, c2, c3 = st.columns([1.0, 1.0, 0.8], gap="large")
            with c1:
                w_qed = st.slider("w_qed（QED）", 0.0, 1.0, 0.70, 0.05, key="small_w_qed")
                w_fsp3 = st.slider("w_fsp3（FracCSP3）", 0.0, 0.5, 0.10, 0.02, key="small_w_fsp3")
            with c2:
                w_pen = st.slider("w_pen（Penalty）", 0.0, 0.6, 0.15, 0.02, key="small_w_pen")
                w_vio = st.slider("w_vio（Violations）", 0.0, 0.4, 0.05, 0.01, key="small_w_vio")
            with c3:
                run_btn = st.button("✅ 运行 Property 打分", width="stretch", key="run_score_small")

            # 运行
            if run_btn:
                with open(in_smi, "w", encoding="utf-8") as f:
                    for s in smiles_list:
                        f.write(s + "\n")

                cmd = [
                    "python", str(SCORE_SCRIPT),
                    "--mode", "property",
                    "-i", str(in_smi),
                    "-o", str(out_csv),
                    "--output_smi", str(out_smi),

                    # Small 权重
                    "--w_qed", str(float(w_qed)),
                    "--w_fsp3", str(float(w_fsp3)),
                    "--w_pen", str(float(w_pen)),
                    "--w_vio", str(float(w_vio)),

                    # ✅ 不做硬过滤：给一个非常宽的默认门槛（仅排除解析失败/极端情况）
                    "--mw_min", "0",
                    "--mw_max", "5000",
                    "--logp_min", "-99",
                    "--logp_max", "99",
                    "--tpsa_min", "0",
                    "--tpsa_max", "500",
                    "--hbd_max", "99",
                    "--hba_max", "99",
                    "--rotb_max", "99",
                    "--qed_min", "0",
                    "--sort_by", "Score",
                ]
                if sort_desc:
                    cmd.append("--descending")

                rc, out, err = run_cmd(cmd)
                if rc != 0:
                    st.error("Property 打分脚本运行失败。")
                    st.code(err or out)
                    return

                df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
                # TopK 截断（在 UI 侧做，避免脚本参数继续膨胀）
                if not df.empty:
                    df = df.head(int(topk))
                    df.to_csv(out_csv, index=False)
                    with open(out_smi, "w", encoding="utf-8") as f:
                        for smi in df["SMILES"].tolist():
                            f.write(smi + "\n")

                st.session_state[f"score_df_{mode_key}"] = df
                st.session_state[f"score_csv_text_{mode_key}"] = out_csv.read_text(encoding="utf-8") if out_csv.exists() else ""
                st.session_state[f"score_smi_text_{mode_key}"] = out_smi.read_text(encoding="utf-8") if out_smi.exists() else ""
                st.success(f"✅ Property 打分完成：输出 {len(df)} 条（TopK={topk}）。")

    # =========================
    # Template 模式 UI + 运行
    # =========================
    else:
        with st.container(border=True):
            st.markdown("### 📚 Template 打分（模板库分布相似）")
            st.caption("上传模板 CSV（大样本），计算模板属性分布；候选分子越接近模板分布得分越高。")

            c1, c2, c3 = st.columns([1.1, 1.2, 0.8], gap="large")

            with c1:
                tpl_csv = st.file_uploader("上传模板库 CSV", type=["csv"], key="tpl_csv")
                tpl_smiles_col = st.text_input("模板 CSV SMILES 列名", value="smiles", key="tpl_smiles_col")

            with c2:
                tpl_features = st.multiselect(
                    "用于模板相似的属性（建议 3~6 个）",
                    ["MW", "LogP", "TPSA", "N_count", "HBD", "HBA", "RotB", "Rings", "FracCSP3"],
                    default=["MW", "LogP", "TPSA", "N_count"],
                    key="tpl_features",
                )
                raw_qed_w = st.slider("raw QED 权重", 0.0, 10.0, 3.0, 0.5, key="tpl_raw_qed_w")

            with c3:
                run_btn = st.button("✅ 运行 Template 打分", width="stretch", key="run_score_tpl")

            st.markdown("---")
            st.markdown("**权重设置（1~5，步长 0.1）**")

            # 初始化/同步权重表（保留用户编辑）
            default_weight_map = {
                "MW": 1.0,
                "LogP": 1.2,
                "TPSA": 1.0,
                "N_count": 1.0,
                "HBD": 1.0,
                "HBA": 1.0,
                "RotB": 1.0,
                "Rings": 1.0,
                "FracCSP3": 1.0,
            }

            if "tpl_weight_df" not in st.session_state:
                st.session_state["tpl_weight_df"] = pd.DataFrame(
                    [{"Feature": f, "Weight": float(default_weight_map.get(f, 1.0))} for f in tpl_features]
                )
            else:
                old_df = st.session_state["tpl_weight_df"].copy()
                old_map = {}
                try:
                    old_map = {str(r["Feature"]): float(r["Weight"]) for _, r in old_df.iterrows()}
                except Exception:
                    old_map = {}
                st.session_state["tpl_weight_df"] = pd.DataFrame(
                    [{"Feature": f, "Weight": float(old_map.get(f, default_weight_map.get(f, 1.0)))} for f in tpl_features]
                )

            edited = st.data_editor(
                st.session_state["tpl_weight_df"],
                hide_index=True,
                width="stretch",
                column_config={
                    "Feature": st.column_config.TextColumn("Feature", disabled=True),
                    "Weight": st.column_config.NumberColumn("Weight", min_value=1.0, max_value=5.0, step=0.1),
                },
                key="tpl_weight_editor",
            )
            st.session_state["tpl_weight_df"] = edited

            # 运行
            if run_btn:
                if tpl_csv is None:
                    st.error("Template 模式需要上传模板库 CSV。")
                    return
                if not tpl_features:
                    st.error("请至少选择 1 个模板相似属性。")
                    return

                with open(in_smi, "w", encoding="utf-8") as f:
                    for s in smiles_list:
                        f.write(s + "\n")

                tpl_path = save_upload(tpl_csv, score_dir / "template_uploads")

                # 从 editor 取权重
                wdf = st.session_state.get("tpl_weight_df")
                weights_dict = {}
                for _, r in wdf.iterrows():
                    feat = str(r["Feature"]).strip()
                    try:
                        w = float(r["Weight"])
                    except Exception:
                        w = 1.0
                    # clamp
                    w = max(1.0, min(5.0, w))
                    weights_dict[feat] = w

                weights_str = weights_dict_to_str(weights_dict)

                cmd = [
                    "python", str(SCORE_SCRIPT),
                    "--mode", "template",
                    "-i", str(in_smi),
                    "-o", str(out_csv),
                    "--output_smi", str(out_smi),

                    "--template_csv", str(tpl_path),
                    "--template_smiles_col", tpl_smiles_col,
                    "--tpl_features", ",".join(tpl_features),
                    "--tpl_weights", weights_str,
                    "--tpl_raw_qed_w", str(float(raw_qed_w)),

                    # ✅ 不做硬过滤：超宽门槛
                    "--mw_min", "0",
                    "--mw_max", "5000",
                    "--logp_min", "-99",
                    "--logp_max", "99",
                    "--tpsa_min", "0",
                    "--tpsa_max", "500",
                    "--hbd_max", "99",
                    "--hba_max", "99",
                    "--rotb_max", "99",
                    "--qed_min", "0",
                    "--sort_by", "Score",
                ]
                if sort_desc:
                    cmd.append("--descending")

                rc, out, err = run_cmd(cmd)
                if rc != 0:
                    st.error("Template 打分脚本运行失败。")
                    st.code(err or out)
                    return

                df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
                if not df.empty:
                    df = df.head(int(topk))
                    df.to_csv(out_csv, index=False)
                    with open(out_smi, "w", encoding="utf-8") as f:
                        for smi in df["SMILES"].tolist():
                            f.write(smi + "\n")

                st.session_state[f"score_df_{mode_key}"] = df
                st.session_state[f"score_csv_text_{mode_key}"] = out_csv.read_text(encoding="utf-8") if out_csv.exists() else ""
                st.session_state[f"score_smi_text_{mode_key}"] = out_smi.read_text(encoding="utf-8") if out_smi.exists() else ""
                st.success(f"✅ Template 打分完成：输出 {len(df)} 条（TopK={topk}）。")

    # =========================
    # 统一：结果展示（表格 + SVG + 下载）
    # =========================
    with st.container(border=True):
        st.markdown("### 📊 结果展示")
        st.caption(f"当前展示模式：{score_mode}（key={mode_key}）")

        df = st.session_state.get(f"score_df_{mode_key}", None)
        if df is None or df.empty:
            st.info("暂无结果。请在上方运行打分。")
            return

        st.dataframe(df, width="stretch", height=420)

        # SVG 预览
        n = min(int(preview_n), len(df))
        mols = []
        for smi in df["SMILES"].head(n).tolist():
            m = Chem.MolFromSmiles(smi)
            if m:
                mols.append(m)

        if mols:
            components.html(
                fragments_to_svg_grid_html(mols, mols_per_row=5, sub_size=(260, 260)),
                height=900,
                scrolling=True,
            )

        csv_text = st.session_state.get(f"score_csv_text_{mode_key}", "")
        smi_text = st.session_state.get(f"score_smi_text_{mode_key}", "")

        dc1, dc2 = st.columns([1, 1], gap="large")
        with dc1:
            st.download_button(
                "⬇️ 下载打分结果（CSV）",
                data=csv_text,
                file_name=f"scored_{mode_key}.csv",
                mime="text/csv",
                width="stretch",
            )
        with dc2:
            st.download_button(
                "⬇️ 下载 TopK 分子（SMI）",
                data=smi_text,
                file_name=f"scored_{mode_key}.smi",
                mime="text/plain",
                width="stretch",
            )

# ==============================
# Router
# ==============================
if selected_module == "分子碎片化":
    render_fragmentation()
elif selected_module == "分子碎片库":
    render_fragment_library()
elif selected_module == "碎片重组":
    render_reassembly()
else:
    render_scoring(score_mode or "Property-Based")


