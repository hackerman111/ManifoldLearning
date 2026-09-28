# HPAO A-operator structure: local state

Status: done. This state is separate from root PLAN.md and
agent-notes/STATE.md; neither was edited for this task.

Established: A_j=sqrt(mass_j)(c_j.T tensor U_j) and A.T A is a sum of local
Kronecker products. On three d10 and three d100 frozen outer0 inputs, all
20/200 columns have resolved full rank at relative singular threshold 1e-8.
Nearest-circulant Gram residuals are 0.269-0.439 / 0.588-0.704; 20/200
eigenvalues are separated at 1e-6 of spectral max. LSMR takes 12-16 / 41-59
iterations. Tiny forward/adjoint/Gram checks are below 3e-15.

Conclusion: no exact circulant/Toeplitz or shared local eigenbasis on the
checked inputs, no low global rank, and no low-degree numerical minimal
polynomial. Polar decomposition exists but does not reduce this ridge solve.
The block/Kronecker factorization is already used by live matrix-free actions.
This is not a universal theorem for future AO states or a full-fit speed claim.

Evidence: report.md, audit.py, results.json. Input hashes and live source
hashes are stored in results.json. Independent tiny explicit-design check
passed below 3e-15; one d100 full-design SVD agreed with Gram singular ratio
within 3e-17. Ruff check/format, six-case JSON assertions and git diff --check
passed. Production files unchanged.
