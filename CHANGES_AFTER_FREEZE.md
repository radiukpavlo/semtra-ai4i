# Changes after the protocol freeze

`PROTOCOL.md` was frozen on 2026-10-23, before any test evaluation, and `outputs/protocol_freeze.sha256` records the
checksums of the frozen files. One file was changed afterwards. The change touches a comment only; no code, setting or
output changed.

| Date | File | Change | Reason |
|------|------|--------|--------|
| 2026-10-23 | `src/semtra_core.py`, line 1 | The upstream commit hash in the provenance comment was replaced with "[masked for double-blind review]" | Double-blind review; the hash is restored in the camera-ready version |

Consequence: `sha256sum -c outputs/protocol_freeze.sha256` reports `src/semtra_core.py: FAILED`; every other file passes
(the CSV after it is downloaded as described in `README.md`).

The code below the two comment lines is unchanged. Its SHA-256, the same before and after the change:

```
tail -n +3 src/semtra_core.py | sha256sum
466a031dbff9ef1694d3952b74860fc8e45ff4d929c76a435abb00723b70805a
```

| `src/semtra_core.py` | SHA-256 of the whole file |
|----------------------|---------------------------|
| Frozen (in `outputs/protocol_freeze.sha256`) | `9a1814755a738bd0c68d112d986df489c9886938960b3d7a8930c8813e6e6f23` |
| Current | `df482ee87ddda24219614b62944e4c40a5af8b5cce093e955fd5f7ee401ed5aa` |

Files are stored byte for byte (`.gitattributes` turns off line-ending conversion), so the checksums hold on any checkout.
