# LiH/STO-3G Stretched Pilot: Grouped BAI ADAPT Gradient Selection

## Scope and assumptions

- System: LiH/STO-3G, all-electron, 4 electrons, 6 spatial orbitals, 12 qubits.
- Geometry: Li--H bond length R = 3.0 Angstrom. This was assumed as stretched but not dissociated.
- Mapping: Jordan--Wigner.
- Reference: RHF determinant occupying spin orbitals 0,1,2,3.
- Pool: spin-conserving UCCSD singles and doubles from the two occupied spatial orbitals to four virtual spatial orbitals.
- This is a first-selection pilot from the RHF reference, not a complete multi-iteration ADAPT-VQE run.

## Electronic structure checks

- RHF energy: -7.710829924325 Ha.
- JW HF expectation: -7.710829924343 Ha.
- FCI energy in the 4-electron space: -7.798843166688 Ha.
- HF error vs FCI: 0.088013242344 Ha.
- Hamiltonian Pauli terms including identity: 631.

## Gradient/pool structure

- Pool size: 92.
- Nonzero gradients > 1e-9: 26.
- Universal commutator Pauli terms: 30196.
- Greedy parent fully commuting groups: 980.
- Largest parent group sizes: [186, 170, 134, 124, 122, 120, 116, 113, 111, 105].

Per-gradient commutator terms: min 286, median 1600.0, max 1648.

Per-gradient FC groups if measured separately: min 28, median 66.0, max 101.

## First selected generator

Winner: index 91 (D 2,3->10,11), gradient 0.243187338287.

Runner-up: index 79 (D 2,3->4,11), gradient 0.146799015287.

Absolute-gradient gap: 0.096388323000.

One-dimensional optimization of winner:

- theta*: -0.268288530421
- E(theta*): -7.744258025696 Ha
- Energy lowering: -0.033428101352 Ha
- Error after one operator vs FCI: 0.054585140992 Ha

## Shot-estimate model

The shot estimates use exact state variances and normal confidence intervals with family-wise delta=0.05. They are planning estimates only; they do not include readout noise, hardware noise, finite-shot Monte Carlo randomness, CNOT cost of FC diagonalization circuits, or optimal group coloring. A context-shot means one circuit shot in one measurement context.

## Baselines and new method

| Method | Result |
|---|---:|
| Naive/no-sharing all-gradient estimate | 37,927,632 context-shots |
| Parent-FC all-gradient estimate | 2,088,380 context-shots |
| Independent BAI | 7,368,553 context-shots |
| New fixed-parent grouped BAI | 1,267,508 context-shots |

With a good-enough tolerance Delta = 0.005 Ha in gradient magnitude:

| Method | Result |
|---|---:|
| Independent BAI, Delta=0.005 | 7,070,522 context-shots |
| New fixed-parent grouped BAI, Delta=0.005 | 1,211,656 context-shots |

## Interpretation

For this first-selection LiH case, the universal parent FC grouping gives a large reduction relative to measuring each gradient independently, because one parent measurement context contributes to many candidate gradients. The approximate fixed-parent grouped BAI improves further by combining shared measurement contexts with active candidate elimination.

The current new-method implementation is deliberately conservative: it fixes parent FC contexts once and only restricts them as BAI eliminates generators. This preserves measurement compatibility and avoids losing earlier samples. The next step is a genuine covariance-aware implementation that chooses the next parent context by maximizing expected reduction in pairwise uncertainty among surviving generators.

## Ambiguities / choices to confirm before production benchmarks

1. The precise stretched LiH geometry. I used R = 3.0 Angstrom.
2. Whether to freeze the Li 1s core and/or taper symmetries. I used all-electron 12-qubit LiH.
3. Which pool should be the main benchmark: spin-orbital UCCSD, qubit pool, qubit-excitation pool, or minimal complete pool.
4. Whether the BAI target should be exact best-gradient identification or good-enough selection within Delta.
5. Whether FC grouping should be greedy sorted insertion, graph coloring, or the exact grouping construction from the paper.
6. Whether shots should be optimized with covariance-aware allocation rather than the current conservative estimates.
