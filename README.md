# CUDA Parallel Algorithms
> 🔬 **Research / Exploratory** — Foundational GPU parallel primitives implemented from scratch as part of graduate coursework in HPC and parallel computing.

Hand-written CUDA implementations of core parallel algorithms — reduction, histogram, scan, and sorting — benchmarked against CPU baselines to understand GPU performance trade-offs.

---

## Algorithms

```
cuda-parallel-algorithms/
├── kernels/
│   ├── parallel_reduction.cu   ← Tree reduction (sum) with warp unrolling
│   ├── histogram.cu            ← Shared memory atomics histogram
│   ├── prefix_scan.cu          ← Exclusive scan (Blelloch algorithm)
│   └── bitonic_sort.cu         ← Parallel bitonic sort
├── benchmarks/
│   └── bench.py                ← CPU vs GPU comparison
└── CMakeLists.txt
```

---

## Kernel Details

### Parallel Reduction
Work-efficient shared memory tree reduction with warp-level unrolling to eliminate `__syncthreads()` in the final 32 threads.

```
Input [N]
  └─ Block 0: sdata[256] → partial sum
  └─ Block 1: sdata[256] → partial sum
       ...
  └─ Final pass: single block reduces all partials
```

| N | CPU (NumPy) | CUDA | Speedup |
|---|---|---|---|
| 1M | 1.21ms | 0.31ms | **3.9×** |
| 16M | 18.4ms | 1.2ms | **15.3×** |

---

### Histogram (Shared Memory Atomics)
Per-block shared histogram merged into global memory, avoiding atomic contention on global memory.

| Strategy | Throughput (GB/s) |
|---|---|
| Global atomics only | 8.2 |
| Shared mem + merge | **41.7** |

---

### Exclusive Prefix Scan (Blelloch)
Work-efficient O(n) scan using up-sweep (reduce) + down-sweep phases entirely in shared memory.

```
Up-sweep:    [1,2,3,4] → [1,3,3,10]
Down-sweep:  [0,1,3,6]  (exclusive)
```

---

### Bitonic Sort
Fully data-parallel sorting network — each comparison-swap is an independent GPU thread. Scales to large arrays on GPU where the parallelism overcomes O(n log² n) work.

| N | CPU (std::sort) | CUDA Bitonic |
|---|---|---|
| 1K | 0.04ms | 0.08ms |
| 64K | 4.1ms | 0.9ms |
| 1M | 89ms | **8.2ms** |

---

## Build & Run

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

./reduction_bench   # Parallel sum
./histogram_bench   # Histogram
./scan_bench        # Prefix scan
./sort_bench        # Bitonic sort

python benchmarks/bench.py  # CPU baseline comparison
```

**Requirements:** CUDA 11+, CMake 3.18+, GCC 9+
