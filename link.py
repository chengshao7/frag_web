import argparse
import re
import random
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, QED

# -----------------------------
# Global Config
# -----------------------------
_dummy_pat = re.compile(r"\[\d+\*\]")  # Matches [1*], [14*], etc.

# -----------------------------
# 1. Cleaning & Canonicalization
# -----------------------------
def clean_fragment_smiles(smi: str) -> str:
    """将同位素标记的 dummy 归一化为 [*]"""
    return _dummy_pat.sub("[*]", smi.strip())

def canonicalize_fragment_smiles(smi: str):
    """标准化 SMILES，去除碎片，确保有效"""
    smi = clean_fragment_smiles(smi)
    if not smi or "." in smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(m, canonical=True)
    except:
        return None

# -----------------------------
# 2. Advanced Endpoint Typing
# -----------------------------
def is_carbonyl_carbon(atom: Chem.Atom) -> bool:
    """判断是否为羰基碳 C(=O)"""
    if atom.GetSymbol() != "C":
        return False
    for b in atom.GetBonds():
        # 必须是双键且连着氧
        if b.GetBondType() == Chem.BondType.DOUBLE:
            other = b.GetOtherAtom(atom)
            if other.GetSymbol() == "O":
                return True
    return False

def is_amide_nitrogen(atom: Chem.Atom) -> bool:
    """判断氮原子是否是酰胺氮 (N-C=O)，此类氮通常亲核性低"""
    if atom.GetSymbol() != "N":
        return False
    for nb in atom.GetNeighbors():
        if is_carbonyl_carbon(nb):
            return True
    return False

def endpoint_type_of_neighbor(nb: Chem.Atom) -> str:
    """
    判断接口类型（dummy atom 的邻接原子）
    优化点：区分了 N_amine (活性) 和 N_amide (惰性)
    """
    sym = nb.GetSymbol()
    
    if sym == "N":
        if is_amide_nitrogen(nb):
            return "N_amide"  # 惰性/低活性
        return "N_amine"      # 活性胺
        
    if sym == "O":
        return "O"
        
    if sym == "S":
        return "S"
        
    if sym == "C":
        if nb.GetIsAromatic():
            return "C_aryl"
        if is_carbonyl_carbon(nb):
            return "C_acyl"   # 酰基/羧基衍生物
        return "C_alkyl"
        
    return "Other"

def get_endpoints(mol: Chem.Mol):
    """
    返回该分子所有可用端点列表
    Result: list of dict {"d": dummy_idx, "n": neighbor_idx, "t": type}
    """
    eps = []
    # 必须通过索引遍历，防止 GetAtoms 顺序问题
    for a in mol.GetAtoms():
        if a.GetSymbol() != "*":
            continue
        d_idx = a.GetIdx()
        neighbors = [n for n in a.GetNeighbors() if n.GetSymbol() != "*"]
        if not neighbors:
            continue
        nb = neighbors[0] # dummy 只有一个邻居
        t_type = endpoint_type_of_neighbor(nb)
        eps.append({"d": d_idx, "n": nb.GetIdx(), "t": t_type})
    return eps

# -----------------------------
# 3. Rule Gating (Compatibility)
# -----------------------------
def allowed_pair(t1: str, t2: str, mode: str = "strict") -> bool:
    """
    决定两个接口是否可以连接
    """
    a, b = sorted([t1, t2])

    # --- 绝对禁止规则 ---
    if "Other" in (a, b): return False
    if "S" in (a, b): return False # 暂不支持硫
    if a == "C_acyl" and b == "C_acyl": return False # 禁止二酮
    if "N_amide" in (a, b): 
        # 严格模式下，禁止在酰胺氮上再接东西（防止形成不稳定的酰亚胺或其他结构）
        # 除非你想做特殊骨架，否则通常 N_amide 是死端
        return False 

    # --- Strict Mode (模拟药物合成) ---
    # 允许：酰胺键、酯键、烷基化、胺化
    strict_allow = {
        ("C_acyl", "N_amine"),  # 酰胺形成 (Amide coupling) - 最重要
        ("C_acyl", "O"),        # 酯形成
        ("C_alkyl", "N_amine"), # 烷基胺 (还原胺化/取代)
        ("C_alkyl", "O"),       # 醚
        ("C_alkyl", "C_alkyl"), # 烷基链延伸
        ("C_aryl", "N_amine"),  # 芳香胺 (Buchwald / SNAr)
        ("C_aryl", "O"),        # 芳香醚
        ("C_aryl", "C_alkyl"),  # 芳基-烷基
    }

    if mode == "strict":
        return (a, b) in strict_allow

    # --- Loose Mode (探索性) ---
    # 允许：Suzuki 偶联 (Ar-Ar), 酰基-烷基 (酮)
    loose_allow_extra = {
        ("C_aryl", "C_aryl"),   # 联苯类
        ("C_acyl", "C_alkyl"),  # 酮
    }
    
    return ((a, b) in strict_allow) or ((a, b) in loose_allow_extra)

# -----------------------------
# 4. Joining Logic (Optimized)
# -----------------------------
def remove_all_dummy_atoms(mol: Chem.Mol) -> Chem.Mol:
    """
    暴力移除所有 [*]，用于生成最终产物（封口）。
    如果产物缺氢，Sanitize 会自动处理（只要价态合理）。
    """
    try:
        # 使用 RWMol 的编辑功能
        rw = Chem.RWMol(mol)
        # 倒序删除，防止索引位移
        dummies = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "*"]
        for idx in sorted(dummies, reverse=True):
            rw.RemoveAtom(idx)
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None

def join_fragments(mol_a: Chem.Mol, mol_b: Chem.Mol, mode: str = "strict"):
    """
    两两拼接核心函数。
    优化点：
    1. 找出所有合法的 (endpoint_A, endpoint_B) 组合。
    2. 随机选择一个组合进行连接（增加多样性）。
    3. 连接后移除所有剩余 dummy atom (两两拼接目的是生成成品)。
    """
    eps_a = get_endpoints(mol_a)
    eps_b = get_endpoints(mol_b)
    
    if not eps_a or not eps_b:
        return None

    # 1. 收集所有可能的连接位点对
    valid_connections = []
    for ea in eps_a:
        for eb in eps_b:
            if allowed_pair(ea["t"], eb["t"], mode=mode):
                valid_connections.append((ea, eb))
    
    if not valid_connections:
        return None

    # 2. 随机选择一种连接方式 (不再总是选第一个)
    # 这让同一个片段对能产生多种异构体（如果它们有多个接口）
    chosen_pair = random.choice(valid_connections)
    ea, eb = chosen_pair

    # 3. 执行物理连接
    # 逻辑：复制 -> 组合 -> 连线 -> 删 dummy
    
    # 标记要反应的 dummy，方便组合后找回
    # 使用 Isotope 标记是一个安全的小技巧
    curr_a = Chem.RWMol(mol_a)
    curr_b = Chem.RWMol(mol_b)
    
    # 获取原子对象并标记
    atom_da = curr_a.GetAtomWithIdx(ea["d"])
    atom_na = curr_a.GetAtomWithIdx(ea["n"])
    atom_da.SetIsotope(999) # 标记 dummy A
    
    atom_db = curr_b.GetAtomWithIdx(eb["d"])
    atom_nb = curr_b.GetAtomWithIdx(eb["n"])
    atom_db.SetIsotope(999) # 标记 dummy B
    
    # 组合
    comb = Chem.CombineMols(curr_a, curr_b)
    rw = Chem.RWMol(comb)
    
    # 寻找刚才标记的原子在 rw 中的新索引
    # 注意：CombineMols 后索引会变，所以必须通过特征（Isotope）找回
    # 这种方法比计算 offset 更不容易出错
    
    final_da_idx = -1
    final_na_idx = -1
    final_db_idx = -1
    final_nb_idx = -1
    
    # 遍历寻找标记点 (Isotope 999) 及其邻居
    # 这一步虽然是 O(N)，但分子很小，速度很快且绝对安全
    matches = []
    for at in rw.GetAtoms():
        if at.GetIsotope() == 999 and at.GetSymbol() == "*":
            matches.append(at.GetIdx())
            
    if len(matches) != 2:
        return None # 异常情况

    # 确定哪个是 A 的 dummy，哪个是 B 的 dummy 并不重要，
    # 重要的是要把它们各自的 neighbor 连起来。
    # 刚才我们标记了 dummy，现在找 dummy 的 neighbor
    
    d1 = rw.GetAtomWithIdx(matches[0])
    n1 = d1.GetNeighbors()[0]
    
    d2 = rw.GetAtomWithIdx(matches[1])
    n2 = d2.GetNeighbors()[0]
    
    # 添加单键
    try:
        rw.AddBond(n1.GetIdx(), n2.GetIdx(), Chem.BondType.SINGLE)
    except:
        return None
        
    # 4. 移除所有 Dummy Atoms (对于两两拼接任务，我们假定这是最后一步)
    # 如果你想保留未反应的接口，请只移除 matches[0] 和 matches[1]
    final_mol = remove_all_dummy_atoms(rw)
    
    if final_mol:
        try:
            AllChem.Compute2DCoords(final_mol)
        except:
            pass
            
    return final_mol

# -----------------------------
# 5. Main Loop
# -----------------------------
def load_fragments_smi(path: Path):
    raw = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            raw.append(line.split()[0])
    return raw
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--max_frags", type=int, default=500)
    ap.add_argument("--max_pairs", type=int, default=10000)
    ap.add_argument("--topk", type=int, default=None, help="Save top K only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", default="strict")
    args = ap.parse_args()

    random.seed(args.seed)
    
    # 1. Load & Clean
    raw_smiles = []
    with open(args.input, "r") as f:
        for line in f:
            if line.strip(): raw_smiles.append(line.split()[0])
            
    uniq = {}
    for s in raw_smiles:
        c = canonicalize_fragment_smiles(s)
        if c: uniq[c] = c
    frags = list(uniq.values())[:args.max_frags]
    
    # Pre-build Mols
    mols = [Chem.MolFromSmiles(s) for s in frags]
    mols = [m for m in mols if m]
    n_mols = len(mols)
    
    if n_mols < 2: return

    # --- 修改点 1: 初始化日志 ---
    # 使用 flush=True 确保主程序能立刻读到这一行
    print(f"INFO: Start generation. Fragments={n_mols}, Target={args.max_pairs}", flush=True)
    
    results = []
    seen = set()
    attempts = 0
    max_attempts = args.max_pairs * 2
    
    # --- 修改点 2: 引入时间库和设置 ---
    import time
    start_time = time.time()
    
    # 设置汇报间隔：每生成 10 个分子打印一行
    # 这样 run_cmd_stream 就能每隔一小会儿收到一行更新
    REPORT_INTERVAL = 10 

    while len(results) < args.max_pairs and attempts < max_attempts:
        attempts += 1
        
        i, j = random.sample(range(n_mols), 2)
        new_mol = join_fragments(mols[i], mols[j], mode=args.mode)
        
        if new_mol is None: 
            # --- 修改点 3: 保活日志 ---
            # 如果连续 5000 次尝试都没结果，打印一行，防止主程序以为卡死了
            if attempts % 5000 == 0:
                print(f"LOG: Searching... Attempts={attempts} Found={len(results)}", flush=True)
            continue
        
        try:
            smi = Chem.MolToSmiles(new_mol, isomericSmiles=True)
            if "." in smi: continue
            
            if smi in seen: continue
            seen.add(smi)
            
            q = QED.qed(new_mol)
            results.append((smi, q))
            
            # --- 修改点 4: 实时进度汇报 (适配 run_cmd_stream) ---
            if len(results) % REPORT_INTERVAL == 0:
                elapsed = time.time() - start_time
                speed = len(results) / elapsed if elapsed > 0 else 0
                percent = (len(results) / args.max_pairs) * 100
                
                # 关键：这里必须输出换行符 (print 默认带)，并且 flush=True
                print(f"PROGRESS: Generated {len(results)}/{args.max_pairs} ({percent:.1f}%) | Speed: {speed:.1f} mol/s", flush=True)
            
        except:
            continue

    # Sort & Filter
    results.sort(key=lambda x: x[1], reverse=True)
    if args.topk and len(results) > args.topk:
        results = results[:args.topk]

    # Save
    with open(args.output, "w") as f:
        # 你的 10.py 不喜欢表头，所以这里不写 Header
        for s, q in results:
            f.write(f"{s}\t{q:.4f}\n")
            
    # print(f"DONE. Saved {len(results)} molecules to {args.output}", flush=True)


if __name__ == "__main__":
    main()
