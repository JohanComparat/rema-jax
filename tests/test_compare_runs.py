"""Comparison of two runs of the finder (rema.validate.compare: tier B of the cosmology study)."""

import numpy as np

from rema.validate import compare as C


def run(rng, n=200):
    ids = np.arange(n) + 1000
    ra, dec = rng.uniform(10, 12, n), rng.uniform(-1, 1, n)
    lam = 10 + 50 * rng.random(n)
    cent = np.stack([ids + 50000, *(np.full(n, -1),) * 4], axis=1)
    return {"SEED_ID": ids, "ID_CENT": cent, "RA": ra, "DEC": dec, "Z_LAMBDA": rng.uniform(0.1, 0.6, n),
            "LAMBDA": lam, "R_LAMBDA": (lam / 100) ** 0.2, "P_CEN": np.tile([[0.8, 0.1, 0.05, 0.05, 0.0]], (n, 1)),
            "MEM_MATCH_ID": np.arange(n, dtype=np.int64)}


def test_rerun_summary():
    rng = np.random.default_rng(2)
    a = run(rng)
    b = {k: v.copy() for k, v in a.items()}
    b["LAMBDA"] = a["LAMBDA"] * 1.01
    b["R_LAMBDA"] = (b["LAMBDA"] / 100) ** 0.2
    order = rng.permutation(len(a["RA"]))
    b = {k: v[order] for k, v in b.items()}
    b["ID_CENT"][:10, 0] = -7                        # centre lost: matched by seed
    b["SEED_ID"][:5] = -1                            # and seed lost too: matched by position
    ia, ib, how = C.match_by_central(a, b)
    assert ia.size == 200 and np.array_equal(a["SEED_ID"][ia][how == 0], b["SEED_ID"][ib][how == 0])
    assert np.bincount(how, minlength=3).tolist() == [190, 5, 5]
    s = C.rerun_summary(a, b, [20, 40], [0.1, 0.35, 0.6], lam_min=20)
    assert abs(s["dlnlam_median"] - np.log(1.01)) < 1e-9 and abs(s["dlnr_median"] - 0.2 * np.log(1.01)) < 1e-9
    assert s["dz_median"] == 0 and s["lost"] == 0 and s["gained"] == 0
    assert 0 < s["centre_changed"] < 0.1 and s["dpcen_median"] == 0
    r = np.asarray(s["ncum_ratio"])
    assert r.shape == (2, 2) and np.all(r >= 1)
    # Members: the same galaxies with probabilities 0.5 and 1 -> overlap 0.5.
    mem_a = {"MEM_MATCH_ID": np.repeat(a["MEM_MATCH_ID"][:3], 2), "ID": np.arange(6), "PMEM": np.full(6, 0.5)}
    mem_b = {"MEM_MATCH_ID": np.repeat(a["MEM_MATCH_ID"][:3] + 1000, 2), "ID": np.arange(6), "PMEM": np.ones(6)}
    ov = C.member_overlap(mem_a, mem_b, a["MEM_MATCH_ID"][:3], a["MEM_MATCH_ID"][:3] + 1000)
    np.testing.assert_allclose(ov, 0.5)
    s2 = C.rerun_summary(a, {k: v[:150] for k, v in b.items()}, [20], [0.1, 0.6], mem_a=mem_a,
                         mem_b={**mem_b, "MEM_MATCH_ID": np.repeat(b["MEM_MATCH_ID"][:3], 2)})
    assert s2["lost"] > 0 and s2["gained"] == 0
