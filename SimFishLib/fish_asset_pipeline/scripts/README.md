# Scripts

This folder will contain the Blender extraction script for milestone 01.

Planned script:

```text
extract_fish_skeleton.py
```

Target command:

```bash
blender -b input.blend --python scripts/extract_fish_skeleton.py -- --output output_dir
```

The script should export:

- `fish_mesh.obj`
- `skeleton.json`
- optional rendered views
- extraction log

Implementation will be added after the documentation is reviewed.
