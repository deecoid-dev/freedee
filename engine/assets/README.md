# Object textures

Drop texture files here to replace the procedural placeholders used by the
demo map. No code changes are needed: the material system checks this folder
before it generates a texture in `engine/materials.py`.

## File names

The file name must match the material name:

| File name          | Used by                          |
| ------------------ | -------------------------------- |
| `crate.png`        | `Crate`                          |
| `wood_wall.png`    | `WoodenWall`                     |
| `stone_wall.png`   | `StoneWall`, `StonePillar`       |
| `brick_wall.png`   | `BrickWall`                      |
| `metal_panel.png`  | `MetalPanel`, `MetalCrate`       |
| `concrete.png`     | `ConcreteBlock`                  |
| `floor_tiles.png`  | `Floor`                          |

Supported extensions:

```
.png .jpg .jpeg .bmp .tga
```

So `stone_wall.jpg` works just as well as `stone_wall.png`. If several files
with the same base name exist, `.png` wins, then `.jpg`, `.jpeg`, `.bmp`,
`.tga`.

## Recommended format

- Square image, ideally 256×256 or 512×512.
- Seamless tiling (the texture repeats across large surfaces).
- Opaque, no colour management or embedded profiles needed.
- PNG is preferred for crisp generated/pattern art; JPG is fine for photos.

Textures are mapped with a box projection and repeat in world metres. The
density per material is set by `UV_SCALES` in `engine/materials.py`; it can
also be overridden per object class with the `uv_scale` attribute.

After replacing a file, restart the demo. The renderer uploads each material
once when it is first drawn.
