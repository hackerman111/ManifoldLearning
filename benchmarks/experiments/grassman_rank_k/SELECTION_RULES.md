# Candidate pool clarification before development analysis

The candidate to validate is one of the four optimized rank policies:
block1, block_full, block_energy, block_cost. The existing methods core,
rank_one, spectral, flat are baselines. block_spectral is a numerical identity
control for existing spectral, so cannot establish a new algorithmic gain.
This pool is fixed before the lead reads any fullfit numerical results.

For each candidate calculate paired fit ratios against core across all six
development scenario/seed pairs. Eligible: all completed finite fits, no extra
failures, candidate median EDR <= core median EDR (pooled, also show each
scenario). Among eligible choose smallest paired median fit ratio; ties by
median EDR then arm name. Success threshold .8 is reported separately; mere
eligibility does not mean a passed success gate. If none eligible, choose
smallest median EDR, then fit ratio/name for explicit negative validation.
The chosen name and development file SHA256 are saved before heldout runs.

No individual heldout result may change selection, tolerances or budgets.
