"""
Inspect a PyTorch checkpoint to extract all pickled global classes/functions.
This helps identify all classes that need to be added to torch.serialization.add_safe_globals()
without needing to retrain the model multiple times.

Usage:
    python scripts/inspect_checkpoint_globals.py path/to/checkpoint.ckpt
"""

import sys
import pickletools
from io import StringIO
from pathlib import Path


def extract_globals_from_checkpoint(checkpoint_path: str) -> set:
    """
    Extract all GLOBAL opcodes (classes/functions) from a checkpoint file.
    
    :param checkpoint_path: Path to the .ckpt or .pt file
    :return: Set of global references in format "module.ClassName"
    """
    globals_found = set()
    
    try:
        with open(checkpoint_path, 'rb') as f:
            # Use pickletools to analyze the pickle opcodes
            output = StringIO()
            pickletools.dis(f, output, memo=None)
            
            # Parse output to find GLOBAL opcodes
            for line in output.getvalue().split('\n'):
                if 'GLOBAL' in line:
                    # Line format: "    GLOBAL   'module' 'ClassName'"
                    parts = line.split("'")
                    if len(parts) >= 4:
                        module = parts[1]
                        class_name = parts[3]
                        globals_found.add(f"{module}.{class_name}")
    except Exception as e:
        print(f"Error reading checkpoint: {e}")
        return set()
    
    return globals_found


def print_safe_globals_imports(globals_found: set) -> None:
    """
    Print the import statements and add_safe_globals call needed for the found globals.
    
    :param globals_found: Set of global references
    """
    if not globals_found:
        print("No globals found in checkpoint.")
        return
    
    print("\n" + "="*80)
    print("FOUND GLOBALS IN CHECKPOINT")
    print("="*80)
    
    # Group by module
    by_module = {}
    for global_ref in sorted(globals_found):
        module, class_name = global_ref.rsplit('.', 1)
        if module not in by_module:
            by_module[module] = []
        by_module[module].append(class_name)
    
    print("\nImport statements:")
    print("-" * 80)
    for module in sorted(by_module.keys()):
        classes = by_module[module]
        if len(classes) == 1:
            print(f"from {module} import {classes[0]}")
        else:
            class_list = ", ".join(sorted(classes))
            print(f"from {module} import (")
            for cls in sorted(classes):
                print(f"    {cls},")
            print(")")
    
    print("\n\ntorch.serialization.add_safe_globals() call:")
    print("-" * 80)
    print("torch.serialization.add_safe_globals([")
    for global_ref in sorted(globals_found):
        _, class_name = global_ref.rsplit('.', 1)
        print(f"    {class_name},")
    print("])")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"Inspecting checkpoint: {checkpoint_path}\n")
    globals_found = extract_globals_from_checkpoint(checkpoint_path)
    print_safe_globals_imports(globals_found)
    
    print(f"Total unique globals found: {len(globals_found)}")
