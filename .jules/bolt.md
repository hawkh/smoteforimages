
## 2023-10-25 - ConstrainedSMOTE _sample_vmf performance bottleneck
**Learning:** The von Mises-Fisher sampling algorithm inside `ConstrainedSMOTE._sample_vmf` was implemented using purely scalar operations inside a nested Python loop (`for idx in range(n_samples): for _attempt in range(100000):`). This O(N) looping was a performance bottleneck.
**Action:** Replaced the Python loops with vectorized NumPy array operations that perform rejection sampling in batches. Handled remaining unconverged samples by maintaining a list of active indices. This improves inference latency significantly and prevents large dataset generation loops from stalling.
