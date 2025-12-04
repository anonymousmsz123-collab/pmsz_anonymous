import numpy as np
import os
from numba import cuda
import math

# ================== 参数 ==================
local_x, local_y, local_z = 512,512,512   # 每个rank的本地块大小
dtype = np.float64
ranks_list = [1,2,4,8,16,32,64,128,256] 
scale = 64.0
seed = 42
output_dir = "./perlin_weak_scaling"
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# CUDA Device functions
# -------------------------
@cuda.jit(device=True, inline=True)
def fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

@cuda.jit(device=True, inline=True)
def lerp(a, b, t):
    return a + t * (b - a)

@cuda.jit(device=True, inline=True)
def grad(hashv, x, y, z):
    h = hashv & 15
    u = x if h < 8 else y
    v = y if h < 4 else (x if h == 12 or h == 14 else z)
    return ((u if (h & 1) == 0 else -u) +
            (v if (h & 2) == 0 else -v))

@cuda.jit(device=True, inline=True)
def perlin(x, y, z, p):
    X = int(math.floor(x)) & 255
    Y = int(math.floor(y)) & 255
    Z = int(math.floor(z)) & 255
    x -= math.floor(x)
    y -= math.floor(y)
    z -= math.floor(z)
    u, v, w = fade(x), fade(y), fade(z)
    A  = p[X] + Y
    AA = p[A] + Z
    AB = p[A + 1] + Z
    B  = p[X + 1] + Y
    BA = p[B] + Z
    BB = p[B + 1] + Z
    return lerp(
        lerp(
            lerp(grad(p[AA], x, y, z),
                 grad(p[BA], x - 1.0, y, z), u),
            lerp(grad(p[AB], x, y - 1.0, z),
                 grad(p[BB], x - 1.0, y - 1.0, z), u), v),
        lerp(
            lerp(grad(p[AA + 1], x, y, z - 1.0),
                 grad(p[BA + 1], x - 1.0, y, z - 1.0), u),
            lerp(grad(p[AB + 1], x, y - 1.0, z - 1.0),
                 grad(p[BB + 1], x - 1.0, y - 1.0, z - 1.0), u), v),
        w)

# -------------------------
# CUDA Kernel
# -------------------------
@cuda.jit
def perlin_kernel(arr, nx, ny, nz, offset_x, offset_y, offset_z, scale, p):
    x, y, z = cuda.grid(3)
    if x < nx and y < ny and z < nz:
        val = perlin((x + offset_x) / scale,
                     (y + offset_y) / scale,
                     (z + offset_z) / scale,
                     p)
        arr[z, y, x] = val

# -------------------------
# Host helpers
# -------------------------
def generate_perlin_block_cuda(offset_x, offset_y, offset_z, nx, ny, nz, scale, p):
    d_arr = cuda.device_array((nz, ny, nx), dtype=dtype)
    threads = (8, 8, 8)
    blocks = (math.ceil(nx/threads[0]),
              math.ceil(ny/threads[1]),
              math.ceil(nz/threads[2]))
    perlin_kernel[blocks, threads](d_arr, nx, ny, nz,
                                   offset_x, offset_y, offset_z,
                                   scale, p)
    return d_arr.copy_to_host()

def find_3d_decomposition(ranks):
    best, min_aspect = None, float("inf")
    for px in range(1, ranks + 1):
        if ranks % px != 0:
            continue
        for py in range(1, ranks // px + 1):
            if (ranks // px) % py != 0:
                continue
            pz = ranks // (px * py)
            dims = sorted([px, py, pz])
            aspect = dims[-1] / dims[0]
            if aspect < min_aspect:
                min_aspect = aspect
                best = (px, py, pz)
    return best

def generate_weak_scaling_samples():
    perm = np.arange(256, dtype=np.int32)
    np.random.seed(seed)
    np.random.shuffle(perm)
    perm = np.concatenate([perm, perm]).astype(np.int32)
    d_perm = cuda.to_device(perm)

    for ranks in ranks_list:
        px, py, pz = find_3d_decomposition(ranks)
        global_x = px * local_x
        global_y = py * local_y
        global_z = pz * local_z
        out_file = os.path.join(output_dir, f"perlin_{global_x}_{global_y}_{global_z}.bin")

        print(f"Generating weak scaling case: ranks={ranks}, global=({global_x},{global_y},{global_z})")

        with open(out_file, "wb") as f:
            # 分块大小，例如一次 256³
            sub_n = 256
            for bz in range(0, global_z, sub_n):
                nz = min(sub_n, global_z - bz)
                for by in range(0, global_y, sub_n):
                    ny = min(sub_n, global_y - by)
                    for bx in range(0, global_x, sub_n):
                        nx = min(sub_n, global_x - bx)
                        block = generate_perlin_block_cuda(bx, by, bz, nx, ny, nz, scale, d_perm)
                        f.write(block.ravel(order="C").tobytes())

        print(f"  ✅ Saved {out_file} ({global_x}×{global_y}×{global_z}, dtype={dtype})")

# -------------------------
if __name__ == "__main__":
    generate_weak_scaling_samples()
