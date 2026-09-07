# -*- coding: utf-8 -*-
"""Walk-sum / loopy-graph convergence numerical verification (offline).

Addresses review point: "Theorem 5.2 only guarantees convergence on trees;
real dependency graphs contain cycles."

Framework (Malioutov et al. 2006, walk-sum):
  Linear belief-update model   x = b + gamma * A x
  whose unique fixed point is  x* = (I - gamma A)^{-1} b
  and equals the walk-sum series  sum_{k>=0} (gamma A)^k b
  whenever gamma * rho(A) < 1 (rho = spectral radius).

This script verifies numerically:
  1. For gamma < 1/rho(A): the truncated series converges to x*, and the
     error decays geometrically at rate gamma*rho(A).
  2. For gamma >= 1/rho(A): the series diverges.
  3. The engine-style iterative propagation with cutoff eps=0.01 reaches a
     solution within a bounded distance of x*; we measure that distance.

Graph families tested:
  - n-cycle rings (n in {3, 5, 10, 20}) with uniform weight w
  - mutual-reinforcement pairs embedded in a chain (the review's
    "risk-controllable <-> supplier-reliable" example)

Usage:
    python scripts/experiment_walksum_verification.py
"""

from __future__ import annotations

import json
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "experiment_walksum_results.json")

CUTOFF = 0.01  # matches engine propagation cutoff
MAX_TERMS = 5000


def build_ring(n: int, w: float) -> np.ndarray:
    """Directed n-cycle with uniform edge weight w."""
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = w
    return A


def build_mutual_pair(chain_len: int, w_cycle: float, w_chain: float) -> np.ndarray:
    """Chain 0->1->...->(k-1) with a mutual-reinforcement cycle on nodes 1,2."""
    n = chain_len
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = w_chain
    A[2, 1] = w_cycle  # mutual reinforcement 1 <-> 2
    A[1, 2] = w_chain
    return A


def spectral_radius(A: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def series_partial_sums(A: np.ndarray, b: np.ndarray, gamma: float, max_terms: int):
    """Return list of (k, ||x_k - x*||) where x_k = sum_{i=0..k} (gamma A)^i b."""
    x_star = np.linalg.solve(np.eye(A.shape[0]) - gamma * A, b)
    errors = []
    x = np.zeros_like(b)
    term = b.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(max_terms):
            x = x + term
            errors.append(float(np.linalg.norm(x - x_star, ord=np.inf)))
            term = gamma * (A @ term)
            if np.linalg.norm(term, ord=np.inf) < 1e-12:
                break
            if not np.isfinite(errors[-1]) or errors[-1] > 1e12:
                break  # diverged; no need to keep summing
    return x_star, errors


def iterative_propagation(A: np.ndarray, b: np.ndarray, gamma: float,
                          x_star: np.ndarray, cutoff: float, max_iter: int = 10000):
    """Engine-style fixed-point iteration with cutoff; returns (error, n_iter)."""
    x = b.copy()
    for it in range(1, max_iter + 1):
        delta = gamma * (A @ x)
        x_new = b + delta
        if np.linalg.norm(x_new - x, ord=np.inf) < cutoff:
            return float(np.linalg.norm(x_new - x_star, ord=np.inf)), it
        x = x_new
    return float(np.linalg.norm(x - x_star, ord=np.inf)), max_iter


def analyze_graph(name: str, A: np.ndarray, gamma_factors=(0.5, 0.9, 1.0, 1.2)) -> dict:
    """Run the full verification battery on one graph."""
    rho = spectral_radius(A)
    n = A.shape[0]
    rng = np.random.default_rng(42)
    b = rng.uniform(0.2, 1.0, size=n)

    out = {"n_nodes": n, "spectral_radius": rho, "gammas": {}}
    for factor in gamma_factors:
        gamma = factor / rho
        key = f"gamma={factor:.2f}/rho"
        entry = {"gamma": gamma, "converges_theory": factor < 1.0}
        try:
            x_star, errors = series_partial_sums(A, b, gamma, MAX_TERMS)
            entry["series_final_error"] = errors[-1]
            entry["series_terms_used"] = len(errors)
            entry["series_error_at_k0"] = errors[0]
            entry["series_error_at_k10"] = errors[min(10, len(errors) - 1)]
            entry["diverges_numerically"] = errors[-1] > 1e6
            if factor < 1.0:
                err_iter, n_iter = iterative_propagation(A, b, gamma, x_star, CUTOFF)
                entry["engine_iter_error"] = err_iter
                entry["engine_iter_steps"] = n_iter
        except np.linalg.LinAlgError:
            entry["series_final_error"] = None
            entry["diverges_numerically"] = True
        out["gammas"][key] = entry
    return out


def main() -> None:
    results: dict = {"graphs": {}, "metadata": {}}

    # 1. Ring graphs of various sizes
    for n in [3, 5, 10, 20]:
        A = build_ring(n, w=0.6)
        results["graphs"][f"ring_n{n}_w0.6"] = analyze_graph(f"ring{n}", A)

    # 2. Mutual-reinforcement pair in a chain (review's example)
    A = build_mutual_pair(chain_len=8, w_cycle=0.7, w_chain=0.6)
    results["graphs"]["mutual_pair_in_chain"] = analyze_graph("mutual", A)

    # 3. Strongly coupled cycle (stress case)
    A = build_ring(5, w=0.95)
    results["graphs"]["ring_n5_w0.95"] = analyze_graph("ring5_strong", A)

    results["metadata"] = {
        "framework": "walk-sum (Malioutov et al. 2006): x* = (I - gamma A)^-1 b = sum_k (gamma A)^k b",
        "convergence_condition": "gamma * rho(A) < 1",
        "cutoff": CUTOFF,
        "max_terms": MAX_TERMS,
        "interpretation": (
            "For gamma < 1/rho(A) the truncated walk-sum series converges "
            "geometrically and the engine's cutoff-based iteration stays within "
            "O(cutoff / (1 - gamma*rho)) of x*; for gamma >= 1/rho(A) both diverge."
        ),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # Console summary
    for gname, g in results["graphs"].items():
        print(f"\n{gname}: rho={g['spectral_radius']:.3f}, n={g['n_nodes']}")
        for k, e in g["gammas"].items():
            conv = "CONV" if e.get("series_final_error", 1e9) < 1e-3 else "DIV "
            eng = (
                f" engine_err={e['engine_iter_error']:.4f} ({e['engine_iter_steps']} steps)"
                if "engine_iter_error" in e else ""
            )
            print(
                f"  {k}: {conv} series_err={e['series_final_error']:.2e} "
                f"terms={e['series_terms_used']}{eng}"
            )
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
