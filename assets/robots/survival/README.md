# Survival scene assets

Quaternius created these assets and released them under CC0 1.0.
The original license files accompany the models. `manifest.json` records each
source URL, SHA-256 digest, and modification.

- `mannequin.glb` comes from Universal Animation Library Standard v3.0.
  Its armor and joint material factors were recolored. Mesh, skeleton, and
  in-place animation buffers retain the original bytes.
- `pistol/` and `cover/` come from Sci-Fi Essentials Standard.
  Their files retain the original bytes.

The renderer scales each crate to the existing cover collider's bounds.
The collider remains a solid box, including the crate's recessed panel details.
Animation does not move the character's collision capsule.

Sources:

- [Universal Animation Library](https://quaternius.itch.io/universal-animation-library)
- [Sci-Fi Essentials Standard](https://opengameart.org/content/sci-fi-essentials-kit)
- [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
