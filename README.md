# SciFi

## Holographic Tree (OpenGL)

This project now uses an OpenGL pipeline (pyglet + moderngl) with instanced geometry for branches and leaves.
Glow is handled in the fragment shader rather than CPU-side line drawing.

### Run

```bash
python3 holographic_tree.py
```

### Controls

- `ESC`: Quit
- `R` or `Space`: Regenerate tree

### Notes

- Requires an OpenGL 3.3-compatible GPU/driver.
- If you need windowed mode, update the `HolographicWindow` constructor in `holographic_tree.py`.
