# pMSz: A Distributed Parallel Algorithm for Correcting Extrema and Morse–Smale Segmentations in Lossy Compression

This repository provides an **anonymous implementation** of **pMSz**, a distributed-memory, GPU-accelerated algorithm for correcting extrema and piecewise linear Morse–Smale segmentations (PLMSS) in error-bounded lossy compression.


### Files

- **`pMSz.cu`**  
  The proposed distributed parallel algorithm described in the paper.

- **`sync_pMSz.cu`**  
  A baseline variant that synchronizes ghost layers after every iteration, used to study synchronization overhead.

- **`naive_MSz.cu`**  
  A direct distributed-memory parallelization of MSz that recomputes integral paths, included as a scalability baseline.

---

## Dependencies

- CUDA Toolkit (`nvcc`)
- MPI with CUDA-aware support (`mpicxx`)
- C++17 compatible compiler
- Zstandard (`zstd`)
- ZFP
- SZ3
- Linux-based HPC system (e.g., SLURM)

---

## Compilation

The code is compiled using `nvcc` with MPI as the host compiler.  
Example compilation command:

```bash
nvcc -ccbin mpicxx -std=c++17 pMSz.cu -o pmsz \
    -lzstd -lzfp -lsz3
```
---

## Running the Code

The program is intended to be executed in a distributed multi-GPU environment using MPI.  
In our experiments, we use `srun` on SLURM-based systems.

### Command Format

```bash
srun -n <num_mpi_ranks> ./pmsz \
    <dataset_path> \
    <dim_x> <dim_y> <dim_z> \
    <abs_error_bound> \
    <compressor_name> \
    <mpi_dim_x> <mpi_dim_y> <mpi_dim_z>
```

### Parameters

- **num_mpi_ranks**  
  Total number of MPI processes. Must satisfy  
  `num_mpi_ranks = mpi_dim_x × mpi_dim_y × mpi_dim_z`.

- **dataset_path**  
  Path to the input 3D scalar field.

- **dim_x dim_y dim_z**  
  Global dataset dimensions.

- **error_bound**  
  Error bound used by the base compressor.

- **compressor_name**  
  Name of the base compressor. Supported options:
  - `sz3`
  - `zfp`

- **mpi_dim_x mpi_dim_y mpi_dim_z**  
  MPI process grid dimensions in x, y, and z directions.

---

## Example

```bash
srun -n 8 ./pmsz data.bin 256 256 256 1e-3 sz3 2 2 2
```

---

## Experimental Setup (Paper)

- **Hardware:** NERSC Perlmutter system with NVIDIA A100 GPUs  
- **Scaling:** Weak scaling up to 128 GPUs; strong scaling up to 16 GPUs  
- **Compressors:** SZ3 and ZFP  
- **Baselines:** MSz (single GPU), naive-MSz, sync-pMSz  
- **Datasets:** Synthetic (Perlin noise) and real-world scientific datasets  
