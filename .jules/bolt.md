## 2025-03-09 - [Vectorized SLERP in ConstrainedSMOTE]
**Learning:** Generating synthetic samples using a for loop to compute SLERP between individual pairs of vectors is extremely slow for high dimensions (512D) and many samples.
**Action:** Always batch numpy operations. Replacing the loop with a vectorized numpy operation, calculating norm, angle, and dot product simultaneously across an entire batch using `np.linalg.norm(..., axis=1)`, yields ~2x speedups. Batch the vector calculations to prevent memory limits (OOM).
