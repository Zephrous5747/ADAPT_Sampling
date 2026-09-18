# Oracle-gradient measurement pilot: H4 and H2O, STO-3G

This is a fallback self-contained run without PySCF/OpenFermion installed. It uses custom STO-3G integrals/Jordan-Wigner bookkeeping. Use as a pilot/debug dataset, not final production numbers.

Assumptions: H2O eq R(OH)=0.9572 A, angle 104.52 deg; H2O stretch R(OH)=1.75 A same angle. H4 eq is a square side 1.0 A; H4 stretch is square side 2.0 A. Pool is standard spin-orbital UCCSD conserving Ne and Sz, not S^2. No tapering.

Shot counts are oracle estimates from exact state variances; no finite-shot Monte Carlo and no experimental variance learning. For H4, FC grouped shot proxies were computed using a no-covariance Pauli-variance approximation. For H2O, only FC context counts and independent-gradient BAI oracle proxies were completed; FC grouped shot proxies timed out in this fallback environment.

| case | state | n_qubits | pool_size | state_energy_hartree | fci_energy_hartree | hamiltonian_pauli_terms | universal_commutator_pauli_terms | parent_fc_groups_greedy | nonzero_gradients | top_gradient_label | top_abs_gradient | top_gap_abs_gradient | oracle_context_shots_naive_no_sharing | oracle_context_shots_parent_fc_all_gradient_proxy | oracle_context_shots_independent_bai_proxy | oracle_context_shots_fixed_parent_grouped_bai_proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H4_square_eq_side1.0A | HF | 8 | 26 | -1.7610750285222343 | -1.9151065134693428 | 93 | 1276 | 356 | 10 | D 2,3 -> 6,7 | 0.295315871454509 | 0.02153937279819862 | 11326894 | 182282619 | 451756 | 45857 |
| H4_square_eq_side1.0A | CISD | 8 | 26 | -1.9104693320864792 | -1.9151065134693428 | 93 | 1276 | 356 | 10 | D 0,1 -> 6,7 | 0.14246687697009947 | 0.11808133649467734 | 267572 | 6066911 | 53265 | 9582 |
| H4_square_stretch_side2.0A | HF | 8 | 26 | -1.5412553101725162 | -1.897849401209091 | 93 | 1276 | 356 | 10 | D 2,3 -> 6,7 | 0.315036062733783 | 0.01664863570721492 | 2732927 | 70335943 | 112099 | 9936 |
| H4_square_stretch_side2.0A | CISD | 8 | 26 | -1.8737253455814007 | -1.897849401209091 | 93 | 1276 | 356 | 10 | D 0,1 -> 6,7 | 0.09896091730373348 | 0.051859056338236745 | 159444 | 7295352 | 19188 | 3441 |
| H2O_eq_R0.9572A_angle104.52 | HF | 14 | 140 | -74.962928 | -75.012404 | 1086 | 81566 | 9874 | 40 | D 4,5 -> 12,13 | 0.30491006 | 0.10109048 | 545516142 | not computed | 16545899 | not computed |
| H2O_eq_R0.9572A_angle104.52 | CISD | 14 | 140 | -75.011701 | -75.012404 | 1086 | 81566 | 9874 | 48 | D 4,5 -> 12,13 | 0.0069254334 | 0.0020642657 | 1286953129357 | not computed | 30149151647 | not computed |
| H2O_stretch_R1.75A_angle104.52 | HF | 14 | 140 | -74.541459 | -74.800924 | 1086 | 81566 | 9874 | 46 | D 6,7 -> 12,13 | 0.38849329 | 0.021855436 | 10932797337 | not computed | 9124527 | not computed |
| H2O_stretch_R1.75A_angle104.52 | CISD | 14 | 140 | -74.764323 | -74.800924 | 1086 | 81566 | 9874 | 48 | D 6,7 -> 12,13 | 0.1068733 | 0.016956377 | 14433032696 | not computed | 92693712 | not computed |
