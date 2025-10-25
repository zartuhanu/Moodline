import os
import shutil
from pathlib import Path

# Source and destination directories
src_dir = Path("lmd_clean")
dst_dir = Path("lmd_deduplicated")
dst_dir.mkdir(parents=True, exist_ok=True)

# To track which songs we've already copied
seen_songs = set()

for root, _, files in os.walk(src_dir):
    for file in files:
        if file.endswith(".mid"):
            # Remove version suffixes like '.1.mid'
            base_name = file
            if '.' in file:
                parts = file.split('.')
                if parts[-2].isdigit():
                    base_name = '.'.join(parts[:-2]) + '.mid'
            # Add unique songs only
            if base_name not in seen_songs:
                seen_songs.add(base_name)
                src_path = Path(root) / file
                relative_path = src_path.relative_to(src_dir)
                dst_subfolder = dst_dir / relative_path.parent
                dst_subfolder.mkdir(parents=True, exist_ok=True)
                dst_path = dst_subfolder / base_name
                shutil.copy(src_path, dst_path)
                print(f"Copied: {src_path} → {dst_path}")