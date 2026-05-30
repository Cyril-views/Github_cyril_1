import json
from pathlib import Path

root_folder = Path(r"c:\Users\clint_7qp7dbl\Downloads\Py-DS-ML-Bootcamp-master (1)\Refactored_Py_DS_ML_Bootcamp-master")

# Find all .ipynb files recursively
ipynb_files = list(root_folder.rglob("*.ipynb"))
print(f"Found {len(ipynb_files)} notebook files")
print("Converting...\n")

success_count = 0
error_count = 0

for ipynb_path in ipynb_files:
    try:
        nb = json.load(ipynb_path.open(encoding='utf-8'))
        py_lines = []

        for cell in nb.get('cells', []):
            ctype = cell.get('cell_type')
            if ctype == 'markdown':
                src = ''.join(cell.get('source', []))
                if src.strip():
                    py_lines.append('"""')
                    py_lines.extend([line.rstrip('\n') for line in src.splitlines()])
                    py_lines.append('"""')
                    py_lines.append('')
            elif ctype == 'code':
                src = ''.join(cell.get('source', []))
                if src.strip():
                    py_lines.extend(src.splitlines())
                    py_lines.append('')

        py_path = ipynb_path.with_suffix('.py')
        py_path.write_text('\n'.join(py_lines), encoding='utf-8')
        print(f"[OK] {ipynb_path.relative_to(root_folder)}")
        success_count += 1
    except Exception as e:
        print(f"[ERROR] {ipynb_path.relative_to(root_folder)}: {e}")
        error_count += 1

print(f"\n{success_count} converted successfully, {error_count} errors")
