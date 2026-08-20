# Vendored: 3Dmol.js

- **Version:** 2.5.5 (latest release as of 2026-08-20)
- **Source:** https://github.com/3dmol/3Dmol.js (published to npm as `3dmol`)
- **File:** `3Dmol-min.js`, taken verbatim from that release's `build/3Dmol-min.js`
  (the plain UMD build -- exposes the `$3Dmol` global once loaded in a
  `<script>` tag; NOT the `.es6-min.js` module build or the `.ui-min.js`
  build, neither of which this project uses).
- **Provenance:** downloaded via `npm pack 3dmol@2.5.5` and verified against
  the shasum published by the npm registry itself
  (`932bff7b1490bf5ef531a36f9af439eae942d7ed`) before being copied here --
  not taken on faith from a CDN.
- **License:** BSD-3-Clause (3Dmol.js itself, Copyright (c) 2014 University
  of Pittsburgh and contributors), which additionally incorporates GLmol
  (dual MIT/LGPL3), Three.js (MIT), and jQuery (MIT). Full text in `LICENSE`
  in this directory, copied verbatim from the same release -- all four are
  permissive and impose no restriction on redistribution or commercial use,
  but all four notices are kept together here since the minified build
  itself only carries a one-line banner comment, not the full text.

## Why this is vendored instead of CDN-linked

`viz.py` embeds this file's contents directly into every generated HTML
viewer, rather than a `<script src="https://.../3Dmol.js">` CDN reference,
so that both generating AND opening a symbro 3D view work with zero network
access -- deliberately the one piece of this pipeline usable by anyone who
just runs `pip install symbro`, no `installation.yaml` entry, no external
tool, no GPU.

## Updating

There's no automated update process by design -- this is a rarely-changing,
integrity-checked asset, not a live dependency. To update: `npm pack
3dmol@<new-version>`, verify the shasum npm itself reports for that
tarball, replace `3Dmol-min.js` and `LICENSE` from the new tarball's
`build/` and root respectively, and update the version/shasum recorded
above.
