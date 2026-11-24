"""Compare two transported task vectors (.pt state-dict deltas): per-key and global relative L2 difference.
Usage: python compare_tvs.py <ref.pt> <new.pt> <out.json>   Prints the global relative difference."""
import json, sys, torch
a = torch.load(sys.argv[1], map_location="cpu"); b = torch.load(sys.argv[2], map_location="cpu")
a = a.get("state_dict", a) if isinstance(a, dict) else a; b = b.get("state_dict", b) if isinstance(b, dict) else b
ka, kb = set(a), set(b)
rows, num, den = {}, 0.0, 0.0
for k in sorted(ka & kb):
    x, y = a[k].double(), b[k].double()
    d = float((x - y).norm()); n = float(x.norm())
    rows[k] = {"rel": d / max(n, 1e-30), "norm_ref": n, "norm_new": float(y.norm())}
    num += d * d; den += n * n
out = {"ref": sys.argv[1], "new": sys.argv[2], "only_ref": sorted(ka - kb), "only_new": sorted(kb - ka),
       "global_rel": (num / den) ** 0.5 if den else None,
       "max_key_rel": max(r["rel"] for r in rows.values()) if rows else None, "per_key": rows}
json.dump(out, open(sys.argv[3], "w"), indent=1)
print(f"global_rel={out['global_rel']:.3e} max_key_rel={out['max_key_rel']:.3e} keys={len(rows)} only_ref={len(out['only_ref'])} only_new={len(out['only_new'])}")
