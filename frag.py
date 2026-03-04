import os
import time
import threading
from functools import wraps
from rdkit import Chem
from rdkit.Chem import BRICS, Recap
import argparse

# -------------------------
# 读取分子函数
# -------------------------
def read_molecules(file_path):
    """read molecules via rdkit
    Parameters
    ----------
    file_path : str
        the molecular file path
    Returns
    -------
    mols : list
        RDKit molecule list
    """
    # 判断文件类型
    file_type = file_path.split('.')[-1].lower()
    if file_type == 'sdf':
        suppl = Chem.SDMolSupplier(file_path)
        mols = [x for x in suppl if x is not None]
    elif file_type == 'smi':
        suppl = Chem.SmilesMolSupplier(file_path, delimiter=' ', titleLine=False, nameColumn=-1)
        mols = [x for x in suppl if x is not None]
    elif file_type == 'mol':
        suppl = Chem.MolSupplier(file_path)
        mols = [x for x in suppl if x is not None]
    elif file_type == 'mol2':
        # 你需要自己实现 Mol2MolSupplier
        suppl = Mol2MolSupplier(file_path)
        mols = [x for x in suppl if x is not None]
    elif file_type == 'pdb':
        from rdkit.Chem import AllChem
        mols = [AllChem.MolFromPDBFile(file_path)]
    else:
        raise ValueError(f'Unsupported file type: {file_type}')
        
    if len(mols) == 0:
        raise ValueError('No molecules found in file!')
        
    return mols

# -------------------------
# Timeout 装饰器
# -------------------------
def timeout_decorator(timeout):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            def target_func():
                result[0] = func(*args, **kwargs)

            thread = threading.Thread(target=target_func)
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                print("Function execution timed out.")
                return None
            else:
                return result[0]

        return wrapper

    return decorator

# -------------------------
# BRICS / RECAP 处理
# -------------------------
@timeout_decorator(timeout=5)
def Brics_frag(mol):
    return BRICS.BRICSDecompose(mol)

@timeout_decorator(timeout=10)
def Recap_frag(mol):
    recap_tree = Recap.RecapDecompose(mol)
    return list(recap_tree.children.keys())

# -------------------------
# MacFrag 调用
# -------------------------
def MacFrac(smi_path, output):
    func = '~/2frag/MacFrag/MacFrag1.py'  # 请改成自己的脚本路径
    os.system(f'python {func} -i {smi_path} -o {output} -maxBlocks 6 -maxSR 8 -asMols False -minFragAtoms 1')

# -------------------------
# Main 处理函数
# -------------------------
def main(smi_path, output, methods):
    base = os.path.splitext(os.path.basename(smi_path))[0]

    if ('Brics' in methods) or ('Recap' in methods):
        start = time.time()
        mols = read_molecules(smi_path)
        print(f'read mols runtime: {time.time() - start:.2f}s')

    # 1️⃣ BRICS
    if 'Brics' in methods:
        start = time.time()
        brics_dir = os.path.join(output, 'brics')
        os.makedirs(brics_dir, exist_ok=True)
        out_file = os.path.join(brics_dir, f'{base}_brics_frag.smi')

        with open(out_file, 'w') as f:
            for mol in mols:
                frags = Brics_frag(mol)
                if frags:
                    f.write('\n'.join(frags) + '\n')
        print(f'brics runtime: {time.time() - start:.2f}s')

    # 2️⃣ RECAP
    if 'Recap' in methods:
        start = time.time()
        recap_dir = os.path.join(output, 'recap')
        os.makedirs(recap_dir, exist_ok=True)
        out_file = os.path.join(recap_dir, f'{base}_recap_frag.smi')

        with open(out_file, 'w') as f:
            for mol in mols:
                frags = Recap_frag(mol)
                if frags:
                    f.write('\n'.join(frags) + '\n')
        print(f'recap runtime: {time.time() - start:.2f}s')

    # 3️⃣ MacFrag
    if 'Macfrag' in methods:
        start = time.time()
        mac_dir = os.path.join(output, 'macfrag')
        os.makedirs(mac_dir, exist_ok=True)

        tmp_out = os.path.join(mac_dir, base)
        os.makedirs(tmp_out, exist_ok=True)

        MacFrac(smi_path, tmp_out)

        src = os.path.join(tmp_out, 'macfrag_frag.smi')
        dst = os.path.join(mac_dir, f'{base}_macfrag_frag.smi')

        if os.path.exists(src):
            os.rename(src, dst)
            os.rmdir(tmp_out)

        print(f'macfrag runtime: {time.time() - start:.2f}s')

# -------------------------
# CLI
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make fragments for smiles file')
    parser.add_argument('-i', '--smi_path', required=True)
    parser.add_argument('-o', '--output', default=os.getcwd())
    parser.add_argument('-m', '--methods', default='Brics', choices=['Brics', 'Recap', 'Macfrag'])
    args = parser.parse_args()

    smi_path = args.smi_path
    output   = args.output
    methods  = [args.methods] if isinstance(args.methods, str) else args.methods

    # -------------------------
    # 如果输入是目录，则批量处理
    # -------------------------
    if os.path.isdir(smi_path):
        files = [f for f in os.listdir(smi_path) if f.endswith('.smi')]
        for f in files:
            full_path = os.path.join(smi_path, f)
            print(f"\n🔹 Processing {f} ...")
            main(full_path, output, methods)
    else:
        main(smi_path, output, methods)

