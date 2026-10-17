# Repair provenance

The first frozen launch in `results/` retained 90 CLI process errors because the runner mapped each frozen NPZ as `path` but did not pass a `--snapshot` argument. No solver arm ran in that launch.
This directory imports the original `fullfit5.jsonl`, `capture.jsonl`, and all captured NPZ files byte-for-byte; the development fits and captures were not rerun. Repaired frozen solves run under the current manifest and are separate from the original failed launch.
Original runner bytes are preserved as `../results/launch_original.py.txt`. The precise change was an added `"snapshot": snapshot["path"]` entry in frozen job construction; the old job object omitted that key. Original runner SHA-256: `0a186bf7628fb6bd6d6cf012cd5cc9b469a55cc181bd287668a5958ee93e4896`.
Fixed runner SHA-256: `0af755d49cd9a8dbb6753dfa290fcbe28a0d07dbade6f5a67bd20eded8dc2474`.
Only the frozen job argument translation in `run.py` changed after the original stages; numerical worker and analysis code, protocol, and captured arrays are unchanged.

Imported artifact SHA-256 values:
- `fullfit5.jsonl`: `8da662cdf4c57b35bd77c2b6ca8159cd8b203b1c58b67d08d39c26321c62df0b`
- `capture.jsonl`: `b347099b3629a7c574a230c70c8450d60f6f38755d365f61b512dfdba638fc38`
- `captures/core_seed0_call0.npz`: `8c8ed363c27628be6e98c0e6469f36ea609ae4c1dcd9336d9a509a23ede7b870`
- `captures/core_seed0_call160.npz`: `a95196658de7fcd071131f9d534d917b13c79077cca441ef5989051444dc29db`
- `captures/core_seed0_call80.npz`: `4100893abb067dd2d8025314514ceda94787c614a6a04fd8ac397277679f56a9`
- `captures/core_seed1_call0.npz`: `ca7ce2a585b0e8e77a736803324511a4caa0ba086db1eac73c0a1e0736f57b21`
- `captures/core_seed1_call160.npz`: `da716996f5902fc2c5589ce5c9bc0d7589c06010cd6bace8ed14cae72e3aa8bf`
- `captures/core_seed1_call80.npz`: `fceaf92f2987f9ed4e866ff05e6ac10a1734537c761cdc6bb9c8fc85a26f57df`
- `captures/core_seed2_call0.npz`: `e49f1eb7eab201a241c01ddd92a1c35f21aab33818161b1dae5a706015543cc1`
- `captures/core_seed2_call160.npz`: `c99c40c8ff5ad2a69bc0d074a91e1a6c02bf5a6770e34e9f7228c7190bf384eb`
- `captures/core_seed2_call80.npz`: `6c0fdf8bde0eb9ec83ed44bdb3f224d0b71ad8cffe93e499654775e7ba9030fe`
- `captures/metric2_seed0_call0.npz`: `8c8ed363c27628be6e98c0e6469f36ea609ae4c1dcd9336d9a509a23ede7b870`
- `captures/metric2_seed0_call160.npz`: `154be9b580c55131cf80475f075cc8d94451ac8a9a683b73eaf2ff9f8cb0f016`
- `captures/metric2_seed0_call80.npz`: `f5af5537b2ca975d11b5e56e8f3f333da9dc6b7c3230ec8baa3c10a5c4efa00f`


Original failed launch rows are also included in `original_launch_errors.jsonl` byte-for-byte (SHA-256 `cc1040cd0a054cf920c01dc7a21c24e03066b3a42d9e71478093c010e61a9e08`; 90 process errors). This keeps the repaired directory's aggregate failure count honest.
