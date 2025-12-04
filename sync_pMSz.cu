#include <mpi.h>
#include <iostream>
#include <queue>
#include <zfp.h>
#include <unordered_map>
#include <bitset>
#include <numeric>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <limits>
#include <atomic>
#include <cstring>
#include <iomanip>
#include <string>
#include "SZ3/api/sz.hpp"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <filesystem>
#include "zstd.h"
#include <zdict.h>
#include <unistd.h>
#include <map>          
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cstdio>

double additional_time = 0.0;
double compression_time = 0.0;
size_t cmpSize = 0;
double maxValue, minValue;
std::uintmax_t storageOverhead = 0;
size_t edit_cnt = 0;
int ite = 0;
int check = 0;
double editsTime = 0;
__device__ bool filtered = false;
std::vector<std::vector<float>> time_counter;
// ===== Simple MPI phase timer (drop-in) =====

#ifndef TX
#define TX 8
#endif
#ifndef TY
#define TY 8
#endif
#ifndef TZ
#define TZ 8
#endif
#ifndef H
#define H 1              
#endif

#ifndef QCAP
#define QCAP 4096
#endif

#ifndef USE_OWNER_RULE
#define USE_OWNER_RULE 1
#endif
template <typename T>
struct SparseEntry {
    int idx;  // flat index
    T value;  // scalar value
};


__device__ __forceinline__ size_t lin3(int x,int y,int z, int W,int Ht){
    return (size_t)x + (size_t)y*W + (size_t)z*W*Ht;
}
__device__ __forceinline__ bool in_owned_region(int gx,int gy,int gz,int ox,int oy,int oz){
    return (gx>=ox && gx<ox+TX) && (gy>=oy && gy<oy+TY) && (gz>=oz && gz<oz+TZ);
}
__device__ __forceinline__ int owned_local_id(int gx,int gy,int gz,int ox,int oy,int oz){
    return (gz-oz)*TY*TX + (gy-oy)*TX + (gx-ox);
}
__device__ __forceinline__ bool owned_by_me(int gx,int gy,int gz,
                                            int W,int Ht,int D){
    
    int obx = gx / TX, oby = gy / TY, obz = gz / TZ;
    return (obx==blockIdx.x && oby==blockIdx.y && obz==blockIdx.z);
}

__device__ __forceinline__ void sbit_set(uint32_t* bits, int id){
    atomicOr(&bits[id>>5], 1u<<(id&31));
}
__device__ __forceinline__ bool sbit_test(uint32_t* bits, int id){
    return ((bits[id>>5] >> (id&31)) & 1u) != 0;
}

__device__ __forceinline__ bool sbit_set_once(uint32_t* bits, int id){
    uint32_t* w = &bits[id>>5];
    uint32_t  m = 1u<<(id&31);
    uint32_t  old = atomicOr(w, m);
    return ( (old & m) == 0 ); 
}

enum Phase {
  P_TOTAL = 0,
  P_READ_INPUT,
  P_EXCHANGE_BEFORE,
  P_GPU_PREP,
  P_KERNEL,
  P_GATHER_SPARSE,
  P_REDUCTIONS,
  P_EXCHANGE_AFTER,
  P_MISC,
  P_COUNT
};

static const char* PHASE_NAME[P_COUNT] = {
  "TOTAL",
  "READ_INPUT",
  "EXCHANGE_BEFORE",
  "GPU_PREP",
  "KERNEL",
  "GATHER_SPARSE",
  "REDUCTIONS",
  "EXCHANGE_AFTER",
  "MISC"
};

struct PhaseTimer {
  double t_start[P_COUNT]{};
  double t_accu[P_COUNT]{};

  inline void begin(Phase p) { t_start[p] = MPI_Wtime(); }
  inline void end(Phase p)   { t_accu[p] += MPI_Wtime() - t_start[p]; }

  void reduce_and_print(MPI_Comm comm, int rank) {
    std::vector<double> maxv(P_COUNT), avgv(P_COUNT);
    for (int p=0;p<P_COUNT;++p) {
      double loc = t_accu[p], gmax=0.0, gsum=0.0;
      MPI_Allreduce(&loc, &gmax, 1, MPI_DOUBLE, MPI_MAX, comm);
      MPI_Allreduce(&loc, &gsum, 1, MPI_DOUBLE, MPI_SUM, comm);
      maxv[p] = gmax;
      avgv[p] = gsum;
    }
    int world=1; MPI_Comm_size(comm,&world);
    for (int p=0;p<P_COUNT;++p) avgv[p] /= world;

    if (rank==0) {
      const double total = std::max(1e-9, maxv[P_TOTAL]);
      printf("\n=== Phase timing (max across ranks, %% of TOTAL) ===\n");
      printf("%-16s %12s %10s %12s\n","PHASE","max(s)","%TOTAL","avg(s)");
      for (int p=0;p<P_COUNT;++p) {
        double pct = 100.0 * maxv[p] / total;
        printf("%-16s %12.6f %9.1f%% %12.6f\n",
               PHASE_NAME[p], maxv[p], pct, avgv[p]);
      }

      printf("\nCSV_HEADERS:");
      for (int p=0;p<P_COUNT;++p) printf("%s%s", (p?",":""), PHASE_NAME[p]);
      printf("\nCSV_MAX_SEC:");
      for (int p=0;p<P_COUNT;++p) printf("%s%.6f", (p?",":""), maxv[p]);
      printf("\n");
    }
  }
};


#define PHASE_BEGIN_COMM(T, P, COMM) do{ MPI_Barrier(COMM); (T).begin(P); }while(0)
#define PHASE_END_COMM(T, P, COMM)   do{ MPI_Barrier(COMM); (T).end(P);   }while(0)


template <typename T> MPI_Datatype get_mpi_datatype();
template <> inline MPI_Datatype get_mpi_datatype<uint8_t>()  { return MPI_UINT8_T; }
template <> inline MPI_Datatype get_mpi_datatype<uint16_t>() { return MPI_UINT16_T; }
template <> inline MPI_Datatype get_mpi_datatype<uint32_t>() { return MPI_UINT32_T; }
template <> inline MPI_Datatype get_mpi_datatype<uint64_t>() { return MPI_UINT64_T; }
template <> inline MPI_Datatype get_mpi_datatype<int8_t>()   { return MPI_INT8_T; }
template <> inline MPI_Datatype get_mpi_datatype<int16_t>()  { return MPI_INT16_T; }
template <> inline MPI_Datatype get_mpi_datatype<int32_t>()  { return MPI_INT32_T; }
template <> inline MPI_Datatype get_mpi_datatype<int64_t>()  { return MPI_INT64_T; }
template <> inline MPI_Datatype get_mpi_datatype<float>()    { return MPI_FLOAT; }
template <> inline MPI_Datatype get_mpi_datatype<double>()   { return MPI_DOUBLE; }

int PX, PY, PZ;
float datatransfer = 0.0;
float finddirection = 0.0;
float getfcp = 0.0;
float fixtime_cp = 0.0;
float packing = 0.0;
float unpacking = 0.0;
float sending = 0.0;
double start_time, end_time, start_time_total;
std::vector<std::map<std::string, double>> times;
std::vector<double> comm_time, comp_time, pack_time, send_time;
int maxNeighbors_host = 14;
__device__ int maxNeighbors = 14;
double *sendbuff_right, *recvbuff_right, 
*sendbuff_left, *recvbuff_left, 
*sendbuff_top, *recvbuff_top, 
*sendbuff_bottom, *recvbuff_bottom,
*sendbuff_ld, *recvbuff_ld,
*sendbuff_ru, *recvbuff_ru;
size_t total_edit_cnt = 0;
size_t *all_max, *all_min;
int *DS_M, *AS_M, *dec_DS_M, *dec_AS_M, *de_direction_as, *de_direction_ds;
int *updated_vertex;
int *or_types;
int *edits;
uint8_t *delta_counter;
__device__ unsigned int count_f_max = 0, count_f_min = 0, edit_count = 0, count_f_dir = 0;
double delta;
int threshold, q;
struct RecvBuffer {
    double* buffer;
    int dx, dy, dz;
    int dir_tag;
};

#define direction_N 14

using namespace std;
__device__ int directions_host[42] = 
{1,0,0,-1,0,0,
0,1,0,0,-1,0,
0,0,1, 0,0,-1,
-1,1,0,1,-1,0, 
-1,0,1,1,0,-1,
0,1,1,0,-1,-1,  
-1,1,1, 1,-1,-1};


// __device__ int directions_host[direction_N + 6] = {0,1,0,0,-1,0,1,0,0,-1,0,0, 0, 0, 1, 0, 0, -1};
__device__ int dx[] = {+1,  0, -1,  0, +1, -1, +1, -1};
__device__ int dy[] = { 0, +1,  0, -1, +1, +1, -1, -1};

int directions_host1[42] = 
{1,0,0,-1,0,0,
0,1,0,0,-1,0,
0,0,1, 0,0,-1,
-1,1,0,1,-1,0, 
-1,0,1,1,0,-1,
0,1,1,0,-1,-1,  
-1,1,1, 1,-1,-1};

#define HALO 1
#define TILE_SIZE 8
#define BLOCK_SIZE 10

#define IDX(x, y, z, dimx, dimy) ((z) * (dimx) * (dimy) + (y) * (dimx) + (x))

const double INVALID_VAL = 1e30;

struct Ghost {
    int global_id = -1;
    int result = -1;
};
    

struct is_nonzero {
  __host__ __device__ bool operator()(const double v) const { return v != 0; }
};

template <typename T>
void compact_edits_dense_to_sparse(const T* d_edits, size_t N,
                                   thrust::device_vector<T>& out_val)
{

    out_val.resize(N);

    
    auto first_idx = thrust::make_counting_iterator<uint32_t>(0);
    auto last_idx  = first_idx + N;

   
    thrust::device_ptr<const T> edits_ptr(d_edits);

    
    auto val_end = thrust::copy_if(
        edits_ptr, edits_ptr + N,       
        edits_ptr,                      
        out_val.begin(),                
        is_nonzero()
    );
    out_val.resize(val_end - out_val.begin());
   
}


__global__ void getlabel_cuda(int *un_sign_ds, int *un_sign_as, 
    int *direction_as, int *direction_ds, int *DS_M, int *AS_M,
    int num, size_t width_host, size_t height_host, size_t depth_host, int type=0){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i>=num){
        return;
    }

    int x = i % width_host;
    int y = (i / width_host) % height_host;
    int z = (i / (width_host * height_host)) % depth_host;
    if (x == 0 || x >= width_host - 1 || y == 0 || y >= height_host - 1 || z == 0 || z >= depth_host - 1) return;
    
    int cur = AS_M[i];
    int next_vertex;
        
    if (cur!=i and AS_M[cur]!=cur){
        
        next_vertex = AS_M[cur];
        if(AS_M[next_vertex] == next_vertex){
            AS_M[i] = next_vertex;
        }
        else if(AS_M[AS_M[next_vertex]] == AS_M[next_vertex]){
            AS_M[i] = AS_M[next_vertex];
        }
        else if(direction_as[i]!=i){
            
            if(AS_M[next_vertex]!=next_vertex){
                AS_M[i] = AS_M[next_vertex];
            }
            else{
                AS_M[i] = next_vertex;
            }

        }
        
        if (AS_M[AS_M[i]] != AS_M[i]){
            *un_sign_as+=1;
        }
    } 

    
    cur = DS_M[i];
    int next_vertex1;
    if (cur!=i and DS_M[cur]!=cur){
        
        next_vertex1 = DS_M[cur];
        if(DS_M[next_vertex1] == next_vertex1){
            DS_M[i] = next_vertex1;
        }
        else if(DS_M[DS_M[next_vertex1]] == DS_M[next_vertex1]){
            DS_M[i] = DS_M[next_vertex1];
        }
        else if(direction_ds[i]!=i){
            
            if(DS_M[next_vertex1]!=next_vertex1){
                DS_M[i] = DS_M[next_vertex1];
            }
            else{
                DS_M[i] = next_vertex1;
            }

        }
        
        if (DS_M[DS_M[i]]!=DS_M[i]){
            *un_sign_ds+=1;
        }
    } 
        
        
    return;

}



__global__ void PathCompression(int *label, size_t total_elements, 
    size_t width_host, size_t height_host, size_t depth_host, int preserveMSS = 1) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_elements) return;
    int x = i % width_host;
    int y = (i / width_host) % height_host;
    int z = (i / (width_host * height_host)) % depth_host;
    if ((x == 0 || x >= width_host - 1 || y == 0 || y >= height_host - 1 || z == 0 || z >= depth_host - 1) && preserveMSS == 1) return;
    
    int cur = label[i];

    while (label[cur] != cur) {
        cur = label[cur];
    }

    label[i] = cur;
}


template <typename T>
void read_subblock(const char* filename,
                   size_t offset_x, size_t offset_y, size_t offset_z,
                   size_t local_x, size_t local_y, size_t local_z,
                   size_t global_x, size_t global_y, size_t global_z,
                   T* local_data) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        perror("fopen failed");
        exit(1);
    }
    
    size_t padded_x = local_x + 2 * HALO;
    size_t padded_y = local_y + 2 * HALO;

    for (size_t z = 0; z < local_z; ++z) {
        for (size_t y = 0; y < local_y; ++y) {
            size_t global_index = (offset_z + z) * global_y * global_x +
                                  (offset_y + y) * global_x +
                                  offset_x;
            long long byte_offset = global_index * sizeof(T);

            if (fseek(f, byte_offset, SEEK_SET) != 0) {
                perror("fseek failed");
                exit(1);
            }

            size_t dst_index = (z + HALO) * padded_y * padded_x +
                               (y + HALO) * padded_x + HALO;

            size_t read_count = fread(&local_data[dst_index], sizeof(T), local_x, f);
            if (read_count != local_x) {
                fprintf(stderr, "fread failed at z=%zu y=%zu: read %zu\n", z, y, read_count);
                exit(1);
            }
        }
    }

    fclose(f);
}

__global__ void init_val(double* arr, double val, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = val;
}

template <typename T>
void exchange_ghost_layers(T* data, int padded_x, int padded_y, int padded_z,
                           int local_x, int local_y, int local_z,
                           MPI_Comm cart_comm, int dims[3], int rank) {
    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);

    MPI_Request requests[200];  // large enough
    int req_idx = 0;
    std::vector<RecvBuffer> recv_buffers;
    MPI_Datatype dt = get_mpi_datatype<T>();
    // printf("rank %d at [%d %d %d] with dimension [%d %d %d]\n", rank, coords[0], coords[1], coords[2], padded_x, padded_y, padded_z);
    for (int k = 0; k < maxNeighbors_host - 2; ++k) {
        int dx = directions_host1[3*k];
        int dy = directions_host1[3*k + 1];
        int dz = directions_host1[3*k + 2];
        // printf("%d %d %d\n", dx, dy, dz);
        if (dx == 0 && dy == 0 && dz == 0) continue; // skip self

        int nbr_coords[3] = {coords[0] + dx, coords[1] + dy, coords[2] + dz};
        // printf("%d %d %d\n", nbr_coords[0], nbr_coords[1], nbr_coords[2]);
        bool neighbor_exists = 
            (nbr_coords[0] >= 0 && nbr_coords[0] < dims[0] &&
             nbr_coords[1] >= 0 && nbr_coords[1] < dims[1] &&
             nbr_coords[2] >= 0 && nbr_coords[2] < dims[2]);

        // (1) Fill halo with INVALID_VAL if neighbor missing
        if (true) {
            if (dx != 0) {
                size_t x_face = (dx == -1) ? 0 : padded_x - 1;
                
                for (size_t z = 0; z < padded_z; z++) {
                    for (size_t y = 0; y < padded_y; y++) {
                        data[z * padded_y * padded_x + y * padded_x + x_face] = INVALID_VAL;
                    }
                }
            }

            if (dy != 0) {
                size_t y_face = (dy == -1) ? 0 : padded_y - 1;
                
                for (size_t z = 0; z < padded_z; z++) {
                    for (size_t x = 0; x < padded_x; x++) {
                        data[z * padded_y * padded_x + y_face * padded_x + x] = INVALID_VAL;
                    }
                }
            }

            if (dz != 0) {
                
                size_t z_face = (dz == -1) ? 0 : padded_z - 1;
                
                for (size_t y = 0; y < padded_y; y++) {
                    for (size_t x = 0; x < padded_x; x++) {
                        data[z_face * padded_y * padded_x + y * padded_x + x] = INVALID_VAL;
                    }
                }
            }
            
            
            
        }
        if(!neighbor_exists) continue;

        // (2) Otherwise real exchange
        int nbr_rank;
        MPI_Cart_rank(cart_comm, nbr_coords, &nbr_rank);

        int dir_tag = std::min(rank, nbr_rank) * 1000 + std::max(rank, nbr_rank);
        // if(dir_tag == 1002) printf("%d %d\n", rank, nbr_rank);
        int send_count = 0;
        T* send_ptr = nullptr;

        // Determine send buffer
        if (dx != 0 && dy == 0 && dz == 0) {
            // x-face
            send_count = local_y * local_z;
            send_ptr = (T*) malloc(send_count * sizeof(T));
            int x_send = (dx == -1) ? HALO : HALO + local_x - 1;
            int idx = 0;
            for (int z = 0; z < local_z; z++)
            for (int y = 0; y < local_y; y++)
                send_ptr[idx++] = data[(z + HALO) * padded_y * padded_x + (y + HALO) * padded_x + x_send];
        }
        else if (dx == 0 && dy != 0 && dz == 0) {
            // y-face
            send_count = local_x * local_z;
            send_ptr = (T*) malloc(send_count * sizeof(T));
            int y_send = (dy == -1) ? HALO : HALO + local_y - 1;
            int idx = 0;
            for (int z = 0; z < local_z; z++){
                for (int x = 0; x < local_x; x++){
                    send_ptr[idx++] = data[(z + HALO) * padded_y * padded_x + y_send * padded_x + (x + HALO)];
                }
            }
            
        }
        else if (dx == 0 && dy == 0 && dz != 0) {
            // z-face
            send_count = local_x * local_y;
            send_ptr = (T*) malloc(send_count * sizeof(T));
            int z_send = (dz == -1) ? HALO : HALO + local_z - 1;
            int idx = 0;
            
            for (int y = 0; y < local_y; y++){
                for (int x = 0; x < local_x; x++){
                    send_ptr[x + y * local_x] = data[z_send * padded_y * padded_x + (y + HALO) * padded_x + (x + HALO)];
                    // if(x + y * local_x == 8318 && rank == 2) printf("rank 2 here: %.17f: [%d %d %d] = %d with tag: %d\n", send_ptr[x + y * local_x], x+ HALO, y+HALO, z_send, z_send * padded_y * padded_x + (y + HALO) * padded_x + (x + HALO), dir_tag);
                }
            }
            
        }
        else if (dx == 1 && dy == -1 && dz == 0) {
            
            
            send_count = local_z;
            send_ptr = (T*) malloc(send_count * sizeof(T));
            int idx = 0;
            for (int z = HALO; z < HALO + local_z; z++){
                send_ptr[z-1] = data[z * padded_y * padded_x + (HALO) * padded_x + (HALO + local_x - 1)];
            }
        }
        else if (dx == -1 && dy == 1 && dz == 0) {
            
            send_count = local_z;
            send_ptr = (T*) malloc(send_count * sizeof(T));
            int idx = 0;
            for (int z = HALO; z < HALO + local_z; z++){
                send_ptr[idx++] = data[z * padded_y * padded_x + (HALO + local_y - 1) * padded_x + (HALO)];
            }
        }

        else if (dx == 0 && dy == -1 && dz == 1) {
           
            send_count = local_x;
            send_ptr = (T*) malloc(send_count * sizeof(T));
            int idx = 0;
            for (int x = HALO; x < HALO + local_x; ++x)
                send_ptr[idx++] = data[(HALO + local_z - 1) * padded_y * padded_x + (HALO) * padded_x + x];
        }

        else if (dx == 0 && dy == 1 && dz == -1) {
            
            send_count = local_x;
            send_ptr = (T*) malloc(send_count * sizeof(T));
            int idx = 0;
            for (int x = HALO; x < HALO + local_x; ++x)
                send_ptr[idx++] = data[HALO * padded_y * padded_x + (HALO + local_y - 1) * padded_x + x];
        }

        else if (dx == 1 && dy == 0 && dz == -1) {
            
            send_count = local_y;
            send_ptr = (T*) malloc(send_count * sizeof(T));
            int idx = 0;
            for (int y = HALO; y < HALO + local_y; ++y)
                send_ptr[idx++] = data[HALO * padded_y * padded_x + y * padded_x + (HALO + local_x - 1)];
        }

        else if (dx == -1 && dy == 0 && dz == 1) {
            
            send_count = local_y;
            send_ptr = (T*) malloc(send_count * sizeof(T));
            int idx = 0;
            for (int y = HALO; y < HALO + local_y; y++){
                send_ptr[idx++] = data[(HALO + local_z - 1) * padded_y * padded_x + y * padded_x + (HALO)];
            }
        }
        else continue;
        // Allocate recv buffer
        T* recv_ptr = (T*) malloc(send_count * sizeof(T));
        

        // Post send/recv
        
        MPI_Isend(send_ptr, send_count, dt, nbr_rank, dir_tag, cart_comm, &requests[req_idx++]);
        MPI_Irecv(recv_ptr, send_count, dt, nbr_rank, dir_tag, cart_comm, &requests[req_idx++]);
        recv_buffers.push_back({recv_ptr, dx, dy, dz, dir_tag});
        
    }

    MPI_Waitall(req_idx, requests, MPI_STATUSES_IGNORE);

        // (3) Scatter received data into halo
    for (auto& rb : recv_buffers) {
        int idx = 0;
        
        if (rb.dx != 0 && rb.dy == 0 && rb.dz == 0) {
            // x-face
            int x_recv = (rb.dx == -1) ? 0 : padded_x - 1;
            for (int z = 0; z < local_z; z++){
                for (int y = 0; y < local_y; y++){
                    
                    data[(z + HALO) * padded_y * padded_x + (y + HALO) * padded_x + x_recv] = rb.buffer[y + z * local_y];
                }  
            }
            
        }
        else if (rb.dx == 0 && rb.dy != 0 && rb.dz == 0) {
            // y-face
            int y_recv = (rb.dy == -1) ? 0 : padded_y - 1;
            for (int z = 0; z < local_z; z++){
                for (int x = 0; x < local_x; x++){
                    
                    data[(z + HALO) * padded_y * padded_x + y_recv * padded_x + (x + HALO)] = rb.buffer[x + z * local_x];
                    
                }
            }

                
        }
        else if (rb.dx == 0 && rb.dy == 0 && rb.dz != 0) {
            // z-face
            int z_recv = (rb.dz == -1) ? 0 : padded_z - 1;
            for (int y = 0; y < local_y; y++){
                for (int x = 0; x < local_x; x++){
                    
                    data[z_recv * padded_y * padded_x + (y + HALO) * padded_x + (x + HALO)] = rb.buffer[x + y * local_x];
                    // if(rank == 0 && z_recv * padded_y * padded_x + (y + HALO) * padded_x + (x + HALO) == 8590) {
                    //     printf("%d at idx %d: %.17f with tag: %d\n", z_recv * padded_y * padded_x + (y + HALO) * padded_x + (x + HALO), x + y * local_x, rb.buffer[x + y * local_x], rb.dir_tag);
                    //     printf("coords: [%d %d %d], padded dimension: [%d %d %d]\n",  (x + HALO), (y + HALO), z_recv, padded_x, padded_y, padded_z );
                    // }
                }
            }
        }
        else if (rb.dx == 1 && rb.dy == -1 && rb.dz == 0) {
            // (x+, y-) corner
            int x_recv = padded_x - 1;
            int y_recv = 0;
            for (int z = 0; z < local_z; z++){
                data[(z + HALO) * padded_y * padded_x + y_recv * padded_x + x_recv] = rb.buffer[idx++];
                // if(rank == 2) printf("rank 2 is receiving %d->Coords: %d %d %d, value is: %.17f ", (z + HALO) * padded_y * padded_x + y_recv * padded_x + x_recv, x_recv, y_recv, z, rb.buffer[idx]);
            }
            // printf("\n");
                
        }
        else if (rb.dx == -1 && rb.dy == 1 && rb.dz == 0) {
            // (x-, y+) corner
            int x_recv = 0;
            int y_recv = padded_y - 1;
            for (int z = 0; z < local_z; z++){
                data[(z + HALO) * padded_y * padded_x + y_recv * padded_x + x_recv] = rb.buffer[z];
            
                // if(rank == 2) printf("tag is: %d at idx: %d rank 2 is receiving %d->Coords: %d %d %d, value is: %.17f\n ",rb.dir_tag, z, (z + HALO) * padded_y * padded_x + y_recv * padded_x + x_recv, x_recv, y_recv, z + HALO, rb.buffer[z]);
            }
            // printf("\n");
                
        }
        else if (rb.dx == 0 && rb.dy == -1 && rb.dz == 1) {
            // (y-, z+) corner
            int y_recv = 0;
            int z_recv = padded_z - 1;
            for (int x = 0; x < local_x; ++x){
                
                data[z_recv * padded_y * padded_x + y_recv * padded_x + (x + HALO)] = rb.buffer[x];
            }
                
        }
        else if (rb.dx == 0 && rb.dy == 1 && rb.dz == -1) {
            // (y+, z-) corner
            int y_recv = padded_y - 1;
            int z_recv = 0;
            for (int x = 0; x < local_x; ++x){
                
                data[z_recv * padded_y * padded_x + y_recv * padded_x + (x + HALO)] = rb.buffer[x];
            }
                
        }
        else if (rb.dx == 1 && rb.dy == 0 && rb.dz == -1) {
            // (x+, z-) corner
            int x_recv = padded_x - 1;
            int z_recv = 0;
            for (int y = 0; y < local_y; ++y){
                
                data[z_recv * padded_y * padded_x + (y + HALO) * padded_x + x_recv] = rb.buffer[y];
            }
        }
        else if (rb.dx == -1 && rb.dy == 0 && rb.dz == 1) {
            // (x-, z+) corner
            int x_recv = 0;
            int z_recv = padded_z - 1;
            for (int y = 0; y < local_y; ++y){
                
                data[z_recv * padded_y * padded_x + (y + HALO) * padded_x + x_recv] = rb.buffer[y];
            }
                
        }

        free(rb.buffer);
    }

}

template <typename T>
__device__ bool islarger_shared(const size_t v, const size_t u, 
                                T value_v1, T value_v2){
    return value_v1 > value_v2 || (value_v1 == value_v2 && v > u);
}

template <typename T>
__device__ bool isless_shared(const size_t v, const size_t u, 
                            T value_v1, T value_v2){
    return value_v1 < value_v2 || (value_v1 == value_v2 && v < u);
}


template <typename T>
__device__ void applyDeltaBuffer_local(uint8_t *delta_counter,
                                T *decp_data, T delta, T *input_data, 
                                T bound, int q, int rank, int ite, 
                                int width, int height, int depth, int *send_flag, size_t tid, bool filtered = false) {
    
    if(abs(input_data[tid]-(decp_data[tid] - delta)) <= bound && delta_counter[tid] < q - 1){
        
        decp_data[tid] -= delta; 
        delta_counter[tid] += 1;
    }
    else{
        delta_counter[tid] = q;
        decp_data[tid] = input_data[tid] - bound;
    }
    
    int x = tid % width;
    int y = (tid / width) % height;
    int z = (tid / (width * height)) % depth;
    if(x <= 1 || x >= width -2 || y <= 1 || y >= height -2 || z <= 1 || z >= depth -2) *send_flag = 1;

      
}


template <typename T>
__global__ void iscriticle(int *DS_M, int *AS_M, int *de_direction_as, int *de_direction_ds, 
                            size_t *all_min, T *input_data, T *decp_data, 
                            int *edits, uint8_t *delta_counter, 
                            size_t width_host, size_t height_host, size_t depth_host,
                            size_t data_size, int *or_types, 
                            size_t global_offset_x, size_t global_offset_y, size_t global_offset_z, 
                            size_t w_temp, size_t h_temp, int rank, int *send_flag, int q, T bound, T delta, int ite, 
                            T uTh, T lTh, int data_type=0, int preserveMSS = 0, int checkFinal = 0){

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i>=data_size) return;
    T data_value = decp_data[i];
    T input_data_value = input_data[i];
    
    
    bool is_maxima = true;
    bool is_minima = true;
    
    
    int x = i % width_host;
    int y = (i / width_host) % height_host;
    int z = (i / (width_host * height_host)) % depth_host;
    if (((x == 0 || x >= width_host - 1 || y == 0 || y >= height_host - 1 || z == 0 || z >= depth_host - 1) || decp_data[i] == INVALID_VAL)){
        if(preserveMSS == 1) {
            DS_M[i] = i;
            AS_M[i] = i;
        }
        return;
    }
    bool fixFlag = false;
    
    size_t global_idx = (global_offset_z + z) * h_temp * w_temp +
                        (global_offset_y + y) * w_temp +
                        (global_offset_x + x);
    
    size_t largest_index = i;
    T largest_value = data_value;
    size_t global_largest_index = i;

    size_t smallest_index = i;
    T smallest_value = data_value;
    size_t global_smallest_index = i;
    
    for (int d = 0; d < maxNeighbors; d++) {
        // for(int k = 0; k < 8; k++){
    
            int dx = directions_host[3 * d];
            int dy = directions_host[3 * d + 1];
            int dz = directions_host[3 * d + 2];
            
            int nx = x + dx;
            int ny = y + dy;
            int nz = z + dz;
            
            size_t neighbor = nx + (ny + nz * height_host) * width_host;
            // int smem_x = tx + (nx - x);
            // int smem_y = ty + (ny - y);
            // int smem_z = tz + (nz - z);
            if (static_cast<int64_t>(nx) < 0 || nx >= width_host || static_cast<int64_t>(ny) < 0 || ny >= height_host || nz < 0 || nz >= depth_host || neighbor >= data_size || (decp_data[neighbor] == INVALID_VAL)) continue;
            T neighbor_value = decp_data[neighbor];
            // uint16_t shared_neighbor_value = smem[smem_z][smem_y][smem_x];
            // if(shared_neighbor_value != neighbor_value) printf("wrong here!\n");
            if (neighbor_value > data_value) {
                is_maxima = false;
            }
            else if (neighbor_value == data_value and neighbor > i) {
                is_maxima = false;
            }
            
            if (neighbor_value < data_value) {
                is_minima = false;
            }
            else if (neighbor_value == data_value and neighbor < i) {
                is_minima = false;
            }

            if(islarger_shared<T>(neighbor, largest_index, neighbor_value, largest_value)){
                largest_index = neighbor;
                largest_value = neighbor_value;
                global_largest_index = (global_offset_z + nz) * h_temp * w_temp +
                            (global_offset_y + ny) * w_temp +
                            (global_offset_x + nx);
            }

            
            if(isless_shared<T>(neighbor, smallest_index, neighbor_value, smallest_value)){
                smallest_index = neighbor;
                smallest_value = neighbor_value;
                global_smallest_index = (global_offset_z + nz) * h_temp * w_temp +
                            (global_offset_y + ny) * w_temp +
                            (global_offset_x + nx);
            }
        // }
        
        // if(data_type == 0 && i == 892 && rank == 0 ) printf("%lu %lu %u %u %u %u\n", neighbor, i, neighbor_value, data_value, input_data[neighbor] , input_data[i]);
    }
    
    
    if (data_type == 1){
        if(is_maxima) {
            if(preserveMSS == 0) {
                or_types[2 * i] = -1;
                or_types[2 * i + 1] = smallest_index;
            }
            else{
                AS_M[i] = i;
                DS_M[i] = smallest_index;
            }   
            
        }
        else if(is_minima) {
            if(preserveMSS == 0) {
                or_types[2 * i] = -2;
                or_types[2 * i + 1] = largest_index;
            }
            else{
                DS_M[i] = i;
                AS_M[i] = largest_index;
            }
            
        }
        else {
            if(preserveMSS == 0) {
                or_types[2 * i] = smallest_index;
                or_types[2 * i + 1] = largest_index;
            }
            else{
                DS_M[i] = smallest_index;
                AS_M[i] = largest_index;
            }
            
        }
        return;
    }
    
    int original_type = or_types[2 * i];
    
    if (data_type == 0) {
    
        if ((is_maxima &&  original_type!= -1) || (!is_maxima && original_type == -1)) {
            count_f_max+=1;
            fixFlag = true;
            // if(rank == 0) printf("value at: %lu %d %.17f %.17f %.17f %.17f %.17f\n", i, or_types[2 * i + 1], data_value, input_data_value, input_data_value-bound, decp_data[or_types[2 * i + 1]], input_data[or_types[2 * i + 1]] - bound);
            if (original_type != -1 && checkFinal == 0) {
                if (atomicCAS(&edits[i], 0, 1) == 0){
                    
                    applyDeltaBuffer_local<T>(delta_counter,
                                       decp_data, delta, input_data,
                                       bound, q, rank, ite,
                                       width_host, height_host, depth_host, send_flag, i);
                }
                
            } else if(checkFinal == 0) {
                if (atomicCAS(&edits[largest_index], 0, 1) == 0) {
                    applyDeltaBuffer_local<T>(delta_counter,
                                    decp_data, delta, input_data,
                                    bound, q, rank, ite,
                                    width_host, height_host, depth_host, send_flag, largest_index, filtered);
                }
                
            }
        }

        if ((is_minima &&  original_type!= -2) || (!is_minima && original_type == -2)) {
            count_f_min+=1;
            fixFlag = true;

            if (original_type != -2 && checkFinal == 0) {
                int original_smallest_index = original_type == -1? or_types[2 * i + 1]:or_types[2 * i];
                // if(rank == 0) printf("value at: %lu %d %u %u %u %u %u\n", i, original_smallest_index, data_value, input_data_value, input_data_value-bound, decp_data[original_smallest_index], input_data[original_smallest_index] - bound);
                if (atomicCAS(&edits[original_smallest_index], 0, 1) == 0) {
                    applyDeltaBuffer_local<T>(delta_counter,
                                       decp_data, delta, input_data,
                                       bound, q, rank, ite,
                                       width_host, height_host, depth_host, send_flag, original_smallest_index);
                }
            } else if(checkFinal == 0){
                if (atomicCAS(&edits[i], 0, 1) == 0) {
                    applyDeltaBuffer_local<T>(delta_counter,
                                    decp_data, delta, input_data,
                                    bound, q, rank, ite,
                                    width_host, height_host, depth_host, send_flag, i, filtered);
                }
                
            }
        }
        // if(preserveMSS == 0) return;
        if(is_minima) {
            de_direction_ds[i] = i;
            de_direction_as[i] = largest_index;
        }
        else if(is_maxima) {
            de_direction_as[i] = i;
            de_direction_ds[i] = smallest_index;
        }

        else {
            de_direction_as[i] = largest_index;
            de_direction_ds[i] = smallest_index;
        }

        if(preserveMSS == 1) {
            DS_M[i] = smallest_index;
            AS_M[i] = largest_index;
        }   
        
        
        
        // check vertex's largest neighbor if not a max;
        if ((is_minima && original_type!= -2) || (!is_minima && original_type == -2) || (is_maxima &&  original_type!= -1) || (!is_maxima && original_type == -1)){
                count_f_dir+=1;
                return;
        }

        int original_largest_index  = or_types[2 * i + 1];
        if(or_types[2 * i] == -1) original_largest_index =i;
        else if(or_types[2 * i] == -2) original_largest_index = or_types[2 * i + 1];

        if(largest_index != original_largest_index){
            count_f_dir+=1;
            // if(rank == 0 && largest_index == 1191898) printf("%lu %lu: %u %u %d: %u %u\n", rank, largest_index, decp_data[largest_index], input_data[largest_index] - bound, original_largest_index, decp_data[original_largest_index], input_data[original_largest_index] - bound);
            if (checkFinal == 0 && atomicCAS(&edits[largest_index], 0, 1) == 0) {
                applyDeltaBuffer_local(delta_counter,
                                decp_data, delta, input_data,
                                bound, q, rank, ite,
                                width_host, height_host, depth_host, send_flag, largest_index, filtered);
            }
            
            
        }

        int original_smallest_index  = or_types[2 * i];
        if(or_types[2 * i] == -1) original_smallest_index = or_types[2 * i + 1];
        else if(or_types[2 * i] == -2) original_smallest_index = i;

        if(smallest_index != original_smallest_index){
            count_f_dir+=1;
            if (checkFinal == 0 && atomicCAS(&edits[original_smallest_index], 0, 1) == 0) {
                applyDeltaBuffer_local(delta_counter,
                                decp_data, delta, input_data,
                                bound, q, rank, ite,
                                width_host, height_host, depth_host, send_flag, original_smallest_index, filtered);
            }
            
        }
        return;
        
    }
}


__global__ void init_edits(size_t data_size, int *edits){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=data_size) return;
    edits[i] = 0;
}


__host__ __device__ size_t get_send_count(int dx, int dy, int dz, 
    size_t local_x, size_t local_y, size_t local_z) {
    int dsum = abs(dx) + abs(dy) + abs(dz);
    if (dsum == 1) {
        if (dx != 0) return 2 * local_y * local_z;
        if (dy != 0) return 2 * local_x * local_z;
        if (dz != 0) return 2 * local_x * local_y;
    } else if (dsum == 2) {
        if (dx == 0) return 2 * local_x;
        if (dy == 0) return 2 * local_y;
        if (dz == 0) return 2 * local_z;
    } else if (dsum == 3) {
        return 0;
    }
    return 0;
}

__global__ void pack_sparse_modified_indices_kernel(
    const int* __restrict__ edits,
    size_t padded_x, size_t padded_y, size_t padded_z,
    size_t width_host, size_t height_host, size_t depth_host,
    int local_x, int local_y, int local_z,
    int offset_x, int offset_y, int offset_z, 
    int* __restrict__ index_buf,  // output: sparse index list
    int* __restrict__ count,      // output: number of entries
    size_t padded_size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= padded_size) return;
    int x = idx % padded_x;
    int y = (idx / padded_x) % padded_y;
    int z = (idx / (padded_x * padded_y)) % padded_x;

    bool in_outer_two_layers =
        (x <= HALO || x >= local_x) ||
        (y <= HALO || y >= local_y) ||
        (z <= HALO || z >= local_z);
    if(!in_outer_two_layers) return;

    int global_x = x - HALO + offset_x;
    int global_y = y - HALO + offset_y;
    int global_z = z - HALO + offset_z;

    if (global_x < 0 || global_x >= width_host ||
        global_y < 0 || global_y >= height_host ||
        global_z < 0 || global_z >= depth_host) return;
    
    size_t gIdx = global_x + global_y * width_host + global_z * width_host * height_host;
    if (edits[idx] == -1) {
        int i = atomicAdd(count, 1);
        index_buf[i] = gIdx;
    }
}

__global__ void unpack_sparse_edits_kernel(
    double* __restrict__ data,
    
    uint8_t* __restrict__ delta_counter,
    int* __restrict__ edits,
    const int* __restrict__ gidx_buf,  // received sparse gIdx
    int gidx_count,
    size_t padded_x, size_t padded_y,
    int offset_x, int offset_y, int offset_z,
    size_t width_host, size_t height_host,
    double delta
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= gidx_count) return;

    size_t gIdx = gidx_buf[i];

    int gx = gIdx % width_host;
    int gy = (gIdx / width_host) % height_host;
    int gz = gIdx / (width_host * height_host);

    int lx = gx - offset_x + HALO;
    int ly = gy - offset_y + HALO;
    int lz = gz - offset_z + HALO;

    size_t local_flat_idx = lz * padded_y * padded_x + ly * padded_x + lx;

    if (edits[local_flat_idx] != -1 && data[local_flat_idx] != INVALID_VAL) {
        delta_counter[local_flat_idx] += 1;
        data[local_flat_idx] -= delta;
    }
}





template <typename T>
__global__ void pack_send_buffer_two_layers_kernel(
    const double* __restrict__ data, 
    int* edits, uint8_t *delta_counter,
    T* __restrict__ sendbuf,
    size_t padded_x, size_t padded_y, size_t padded_z,
    size_t width_host, size_t height_host, size_t depth_host,
    size_t local_x, size_t local_y, size_t local_z,
    int offset_x, int offset_y, int offset_z,
    int dx, int dy, int dz, int rank, int *count)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

   
    size_t send_count = get_send_count(dx, dy, dz, local_x, local_y, local_z);
    size_t total_send_count = send_count;
    if (idx >= total_send_count) return;

    size_t i = idx / 2;       // Which face index (z/y/x combined)
    bool is_first_layer = (idx % 2 == 0);  // Select layer 0 or layer 1

    size_t x = 0, y = 0, z = 0;
    size_t write_index = 0;

    if (dx != 0 && dy == 0 && dz == 0) {
        z = HALO + i / local_y;
        y = HALO + i % local_y;
        x = (dx == -1) ? (HALO - (is_first_layer ? 0 : 1))
                       : (local_x + (is_first_layer ? 0 : 1));
        
    } else if (dy != 0 && dx == 0 && dz == 0) {
        z = HALO + i / local_x;
        x = HALO + i % local_x;
        y = (dy == -1) ? (HALO - (is_first_layer ? 0 : 1))
                       : (local_y + (is_first_layer ? 0 : 1));

    } else if (dz != 0 && dx == 0 && dy == 0) {
        y = HALO + i / local_x;
        x = HALO + i % local_x;
        z = (dz == -1) ? (HALO - (is_first_layer ? 0 : 1)) : (local_z + (is_first_layer ? 0 : 1));
    } else if (dz == 0 && dx != 0 && dy != 0) {
        x = dx == -1 ? (HALO - (is_first_layer ? 0 : 1))
                     : (local_x + (is_first_layer ? 0 : 1));
        y = dy == -1 ? (HALO - (is_first_layer ? 0 : 1))
                     : local_y + (is_first_layer ? 0 : 1);
        z = HALO + i;
    }

    else if (dz != 0 && dx != 0 && dy == 0) {
        x = (dx == -1) ? (HALO - (is_first_layer ? 0 : 1))
                     : (local_x + (is_first_layer ? 0 : 1));
        y = HALO + i;
        z = (dz == -1) ? (HALO - (is_first_layer ? 0 : 1)) : (local_z + (is_first_layer ? 0 : 1));
    }

    else if (dz != 0 && dx == 0 && dy != 0) {
        x = HALO + i;
        y = (dy == -1) ? (HALO - (is_first_layer ? 0 : 1))
                     : local_y + (is_first_layer ? 0 : 1);
        z = (dz == -1) ? (HALO - (is_first_layer ? 0 : 1)) : (local_z + (is_first_layer ? 0 : 1));
    }
    
    
    size_t global_idx = (z) * padded_y * padded_x + (y) * padded_x + x;
    int global_x = x - HALO + offset_x;
    int global_y = y - HALO + offset_y;
    int global_z = z - HALO + offset_z;

    if (global_x < 0 || global_x >= width_host ||
        global_y < 0 || global_y >= height_host ||
        global_z < 0 || global_z >= depth_host) return;
    
    size_t gIdx = global_x + global_y * width_host + global_z * width_host * height_host;
    if(data[global_idx] == INVALID_VAL) return;
    
    if(delta_counter[global_idx] != 0){
       
        sendbuf[idx] = delta_counter[global_idx];
    }
    
}


template <typename T>
__global__ void unpack_recv_buffer_two_layers_kernel(
    double* __restrict__ data,
    double* input_data, double bound,
    uint8_t* delta_counter, int *edits,
    const T* __restrict__ recvbuf,
    size_t padded_x, size_t padded_y, size_t padded_z,
    size_t local_x, size_t local_y, size_t local_z,
    double delta,
    int dx, int dy, int dz, int rank)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    size_t total_recv_count = get_send_count(dx, dy, dz, local_x, local_y, local_z);
    if (idx >= total_recv_count || recvbuf[idx] == 0 ) return;
    
    size_t i = idx / 2;
    bool is_first_layer = (idx % 2 == 0);

    size_t x = 0, y = 0, z = 0;

    if (dx != 0 && dy == 0 && dz == 0) {
    
        z = HALO + i / local_y;
        y = HALO + i % local_y;
        x = (dx == -1) ? (0 + (is_first_layer ? 0 : 1))                  // left ghost
                       : (local_x + (is_first_layer ? 1 : 0));   // right ghost

    } else if (dy != 0 && dx == 0 && dz == 0) {
        z = HALO + i / local_x;
        x = HALO + i % local_x;
        y = (dy == -1) ? (0 + (is_first_layer ? 0 : 1))                  // front ghost
                       : (local_y + (is_first_layer ? 1 : 0));   // back ghost

    } else if (dz != 0 && dx == 0 && dy == 0) {
        y = HALO + i / local_x;
        x = HALO + i % local_x;
        z = (dz == -1) ? (0 + (is_first_layer ? 0 : 1))                  // bottom ghost
                       : (local_z + (is_first_layer ? 1 : 0));   // top ghost
    }

    else if (dz == 0 && dx != 0 && dy != 0) {
        x = dx == -1 ? (0 + (is_first_layer ? 0 : 1))
                     : (local_x + (is_first_layer ? 1 : 0));
        y = dy == -1 ? (0 + (is_first_layer ? 0 : 1))
                     : local_y + (is_first_layer ? 1 : 0);
        z = HALO + i;
    }

    else if (dz != 0 && dx != 0 && dy == 0) {
        x = (dx == -1) ? (0 + (is_first_layer ? 0 : 1))
                     : (local_x + (is_first_layer ? 1 : 0));
        y = HALO + i;
        z = (dz == -1) ? (0 + (is_first_layer ? 0 : 1)) : (local_z + (is_first_layer ? 1 : 0));
    }

    else if (dz != 0 && dx == 0 && dy != 0) {
        x = HALO + i;
        y = (dy == -1) ? (0 + (is_first_layer ? 0 : 1))
                     : local_y + (is_first_layer ? 1 : 0);
        z = (dz == -1) ? (0 + (is_first_layer ? 0 : 1)) : (local_z + (is_first_layer ? 1 : 0));
    }
    

    size_t global_idx = z * padded_y * padded_x + y * padded_x + x;
   
    double currentV = data[global_idx];
    
    T currentDelta = delta_counter[global_idx];
    T recvDelta = recvbuf[idx];
    
    if(data[global_idx] != INVALID_VAL && currentDelta < recvDelta && currentDelta >= 0 && recvDelta > 0){
        delta_counter[global_idx] = recvDelta;
        if(recvDelta < 5) {
            data[global_idx] -= delta * (recvDelta - currentDelta);
        }
        else data[global_idx] = input_data[global_idx] - bound;
    }
}

template <typename T>
__global__ void fill_invalid_kernel(T* buf, size_t n, T val) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        buf[idx] = val;
    }
}


template <typename T>
void exchange_ghost_layers_cuda_aware(double* data, uint8_t* delta_counter, double *input_data_host, double bound, int* edits, 
                                      size_t padded_x, size_t padded_y, size_t padded_z,
                                      size_t width_host, size_t height_host, size_t depth_host,
                                      size_t local_x, size_t local_y, size_t local_z, 

                                      int offset_x, int offset_y, int offset_z, double delta, 
                                      MPI_Comm cart_comm, int dims[3], int rank, int send_flag, int ite) {
    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);

    MPI_Request requests[200];
    int req_idx = 0;

    dim3 blockSize(256);
    std::vector<T*> sendbufs;
    std::vector<T*> recvbufs;
    std::vector<int> dxs, dys, dzs;
    std::vector<size_t> send_counts;
    size_t padded_size = padded_x * padded_y *padded_z;
    // start_time = MPI_Wtime();
    
    MPI_Datatype mpi_dtype = get_mpi_datatype<T>();
    for (int k = 0; k < maxNeighbors_host - 2; ++k) {
        int dx = directions_host1[3 * k], dy = directions_host1[3 * k + 1], dz = directions_host1[3 * k + 2];
        if (dx == 0 && dy == 0 && dz == 0) continue;

        int nbr_coords[3] = {coords[0] + dx, coords[1] + dy, coords[2] + dz};
        bool neighbor_exists =
            (nbr_coords[0] >= 0 && nbr_coords[0] < dims[0] &&
             nbr_coords[1] >= 0 && nbr_coords[1] < dims[1] &&
             nbr_coords[2] >= 0 && nbr_coords[2] < dims[2]);

        if (!neighbor_exists) continue;

        int nbr_rank;
        MPI_Cart_rank(cart_comm, nbr_coords, &nbr_rank);

        // === Step 1: Exchange send_flag with neighbor ===
        int my_flag = send_flag;
        int nbr_flag = 0;

        int dir_tag = std::min(rank, nbr_rank) * 1000 + std::max(rank, nbr_rank);
        MPI_Sendrecv(&my_flag, 1, MPI_INT, nbr_rank, dir_tag,
             &nbr_flag, 1, MPI_INT, nbr_rank, dir_tag,
             cart_comm, MPI_STATUS_IGNORE);
        
        bool do_send = (my_flag == 1);
        bool do_recv = (nbr_flag == 1);
        
        if(check==1){
            do_send = true;
            do_recv = true;
        }
        
        // === Step 2: If either side needs communication, continue
        if (!do_send && !do_recv) {
            
            sendbufs.push_back(nullptr);
            recvbufs.push_back(nullptr);
            dxs.push_back(0); dys.push_back(0); dzs.push_back(0);
            send_counts.push_back(0);
            requests[req_idx++] = MPI_REQUEST_NULL;
            requests[req_idx++] = MPI_REQUEST_NULL;
            continue;
        }
        
        dxs.push_back(dx); dys.push_back(dy); dzs.push_back(dz);
        size_t send_count = get_send_count(dx, dy, dz, local_x, local_y, local_z); // maximum padded bound
        T* sendbuf = nullptr;
        T* recvbuf = nullptr;
        
        int h_count = 0;
        if (do_send) {
            cudaMalloc(&sendbuf, send_count * sizeof(T));
            cudaMemset(sendbuf, 0, send_count * sizeof(T));

            int* count_dev = nullptr;
            cudaMalloc(&count_dev, sizeof(int));
            cudaMemset(count_dev, 0, sizeof(int));

            dim3 grid((send_count + blockSize.x - 1) / blockSize.x);
            pack_send_buffer_two_layers_kernel<T><<<grid, blockSize>>>(
                data, edits, delta_counter, sendbuf, padded_x, padded_y, padded_z,
                width_host, height_host, depth_host,
                local_x, local_y, local_z, offset_x, offset_y, offset_z,
                dx, dy, dz, rank, count_dev);

            cudaMemcpy(&h_count, count_dev, sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(count_dev);
            
            h_count = send_count;
            
        }
        h_count = send_count;
        // Step 4: Exchange h_count with neighbor
        int send_actual_count = do_send ? h_count : 0;
        int recv_actual_count = 0;
        int tag_count = dir_tag + 10000;

        MPI_Sendrecv(&send_actual_count, 1, MPI_INT, nbr_rank, tag_count,
                    &recv_actual_count, 1, MPI_INT, nbr_rank, tag_count,
                    cart_comm, MPI_STATUS_IGNORE);
        
        if (do_recv) {
            cudaMalloc(&recvbuf, recv_actual_count * sizeof(T));
            cudaMemset(recvbuf, 0, recv_actual_count * sizeof(T));
        }

        sendbufs.push_back(sendbuf);
        recvbufs.push_back(recvbuf);
        send_counts.push_back(send_count); // for bookkeeping

        // Step 6: MPI Isend / Irecv
        if (do_send) {
            MPI_Isend(sendbuf, send_count, mpi_dtype, nbr_rank, dir_tag, cart_comm, &requests[req_idx++]);
        } else {
            requests[req_idx++] = MPI_REQUEST_NULL;
        }

        if (do_recv) {
            MPI_Irecv(recvbuf, send_count, mpi_dtype, nbr_rank, dir_tag, cart_comm, &requests[req_idx++]);
        } else {
            requests[req_idx++] = MPI_REQUEST_NULL;
        }
    }

    // Wait for all communication to finish
    

    MPI_Waitall(req_idx, requests, MPI_STATUSES_IGNORE);
    

    cudaDeviceSynchronize();
    // end_time = MPI_Wtime();
    // sending += end_time - start_time;

    // Unpack received buffers
    for (size_t i = 0; i < recvbufs.size(); ++i) {
        if (recvbufs[i] != nullptr) {
            size_t send_count = send_counts[i];
            int dx = dxs[i], dy = dys[i], dz = dzs[i];
            if(dx == 0 && dy == 0 && dz == 0) continue;
            dim3 gridSize_unpack((send_count + blockSize.x - 1) / blockSize.x);
                                                             
           
            unpack_recv_buffer_two_layers_kernel<T><<<gridSize_unpack, blockSize>>>(data, input_data_host, bound, delta_counter, edits, recvbufs[i], padded_x, padded_y, padded_z,
                                                         local_x, local_y, local_z, delta, dx, dy, dz, rank);
        }
    }
    cudaDeviceSynchronize();

    // Free buffers
    for (size_t i = 0; i < recvbufs.size(); ++i) {
        if (sendbufs[i]) cudaFree(sendbufs[i]);
        if (recvbufs[i]) cudaFree(recvbufs[i]);
    }
}





__global__ void pack_modified_sparse_kernel(
    const int* __restrict__ edits,
    const double* __restrict__ data,
    size_t padded_x, size_t padded_y, size_t padded_z,
    size_t width_host, size_t height_host,
    int offset_x, int offset_y, int offset_z,
    int* __restrict__ index_buf,
    int* __restrict__ count) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t padded_size = padded_x * padded_y * padded_z;
    if (idx >= padded_size) return;

    int x = idx % padded_x;
    int y = (idx / padded_x) % padded_y;
    int z = (idx / (padded_x * padded_y)) % padded_z;

    // Ghost or outermost layer
    bool is_boundary = (x <= HALO || x >= padded_x - HALO - 1 ||
                        y <= HALO || y >= padded_y - HALO - 1 ||
                        z <= HALO || z >= padded_z - HALO - 1);
    if (!is_boundary) return;
    if (edits[idx] != -1) return;

    int global_x = x - HALO + offset_x;
    int global_y = y - HALO + offset_y;
    int global_z = z - HALO + offset_z;
    size_t gIdx = global_x + global_y * width_host + global_z * width_host * height_host;

    int i = atomicAdd(count, 1);
    index_buf[i] = static_cast<int>(gIdx);
}

__global__ void unpack_sparse_modified_kernel(
    double* __restrict__ data,
    uint8_t* __restrict__ delta_counter,
    int* __restrict__ edits,
    const int* __restrict__ index_buf,
    int global_size,
    double delta,
    size_t width_host, size_t height_host, size_t depth_host,
    int offset_x, int offset_y, int offset_z,
    size_t padded_x, size_t padded_y, size_t padded_z) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= global_size) return;

    int gIdx = index_buf[idx];
    int gz = (gIdx / (width_host * height_host)) % depth_host;
    int gy = (gIdx % (width_host * height_host)) / width_host;
    int gx = gIdx % width_host;

    int lx = gx - offset_x + HALO;
    int ly = gy - offset_y + HALO;
    int lz = gz - offset_z + HALO;

    if (lx < 0 || ly < 0 || lz < 0 ||
        lx >= padded_x || ly >= padded_y || lz >= padded_z) return;

    size_t lid = lx + ly * padded_x + lz * padded_x * padded_y;
    if (edits[lid] == -1) {
        data[lid] -= delta;
        delta_counter[lid] += 1;
    }
}

template <typename T>
void compressLocalDataZFP(const std::string file_path, std::string cpfilename, const std::string filename,
    const T *input_data_host, T *&decp_data_host, size_t width_host, size_t height_host, size_t depth_host,
    double tolerance, std::string decpfilename, int rank, int processData = 0) 
{
    size_t data_size = width_host * height_host * depth_host;
    std::cout << "sub datasize: " << data_size << std::endl;

    // --- ZFP setup ---
    zfp_type type = (std::is_same<T, double>::value) ? zfp_type_double : zfp_type_float;
    zfp_field* field = zfp_field_3d(const_cast<T*>(input_data_host), type, width_host, height_host, depth_host);

    zfp_stream* zfp = zfp_stream_open(nullptr);
    zfp_stream_set_accuracy(zfp, tolerance); 

    size_t bufsize = zfp_stream_maximum_size(zfp, field);
    void* buffer = malloc(bufsize);
    if (!buffer) {
        std::cerr << "Failed to allocate buffer" << std::endl;
        return;
    }
    bitstream* stream = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    // --- Compression ---
    auto start = std::chrono::high_resolution_clock::now();
    size_t zfpsize = zfp_compress(zfp, field);
    auto end = std::chrono::high_resolution_clock::now();

    if (zfpsize == 0) {
        std::cerr << "Compression failed!" << std::endl;
        return;
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double compression_time = double(duration.count()) / 1000.0;
    std::cout << "Compression time: " << compression_time << " s, compressed size = " << zfpsize << " bytes" << std::endl;

    // Save compressed data if needed
    if (processData != 0) {
        std::ofstream outFile1(cpfilename, std::ios::binary);
        if (!outFile1) {
            std::cerr << "Failed to open file for writing compressed data." << std::endl;
        } else {
            outFile1.write(reinterpret_cast<const char*>(buffer), zfpsize);
            outFile1.close();
            std::cout << "Compressed data saved to " << cpfilename << std::endl;
        }
    }

    // --- Decompression ---
    decp_data_host = new T[data_size];
    zfp_field* field_decomp = zfp_field_3d(decp_data_host, type, width_host, height_host, depth_host);

    stream_rewind(stream);
    start = std::chrono::high_resolution_clock::now();
    if (!zfp_decompress(zfp, field_decomp)) {
        std::cerr << "Decompression failed!" << std::endl;
        return;
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Decompression time: " << double(duration.count()) / 1000.0 << " s" << std::endl;

    // Save decompressed data if needed
    if (processData != 0) {
        std::ofstream outFile(decpfilename, std::ios::binary);
        if (!outFile) {
            std::cerr << "Error: Cannot open output file." << std::endl;
        } else {
            outFile.write(reinterpret_cast<const char*>(decp_data_host), data_size * sizeof(T));
            outFile.close();
            std::cout << "Decompressed data saved to " << decpfilename << std::endl;
        }
    }

    // --- Compression Ratio ---
    std::uintmax_t original_dataSize = std::filesystem::file_size(file_path);
    double cr = double(original_dataSize) / double(zfpsize);
    std::cout << "Data read, compressed, and decompressed successfully: CR = " << cr << std::endl;

    // --- Cleanup ---
    zfp_field_free(field);
    zfp_field_free(field_decomp);
    zfp_stream_close(zfp);
    stream_close(stream);
    free(buffer);
}


template <typename T>
void compressLocalDataZFP_coreOnly(const std::string file_path,
    std::string cpfilename, const std::string filename,
    const T *input_data_host, T *&decp_data_host,
    size_t width_host, size_t height_host, size_t depth_host,  // full size with ghost
    size_t core_x, size_t core_y, size_t core_z,               // core region size                                                // ghost layer size
    double tolerance, std::string decpfilename,
    int rank, int processData = 0)
{
    size_t full_size = width_host * height_host * depth_host;
    size_t core_size = core_x * core_y * core_z;
    std::cout << "Full block size: " << full_size << ", core size: " << core_size << std::endl;

    T* core_buffer = new T[core_size];
    for (size_t z = 0; z < core_z; z++) {
        for (size_t y = 0; y < core_y; y++) {
            const T* src = input_data_host +
                (z + HALO) * width_host * height_host +
                (y + HALO) * width_host +
                HALO;
            T* dst = core_buffer + (z * core_y + y) * core_x;
            std::memcpy(dst, src, core_x * sizeof(T));
        }
    }


    zfp_type type = (std::is_same<T, double>::value) ? zfp_type_double : zfp_type_float;
    zfp_field* field = zfp_field_3d(core_buffer, type, core_x, core_y, core_z);

    zfp_stream* zfp = zfp_stream_open(nullptr);
    zfp_stream_set_accuracy(zfp, tolerance);

    size_t bufsize = zfp_stream_maximum_size(zfp, field);
    void* buffer = malloc(bufsize);
    if (!buffer) { std::cerr << "Failed to allocate buffer" << std::endl; return; }

    bitstream* stream = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    auto start = std::chrono::high_resolution_clock::now();
    size_t zfpsize = zfp_compress(zfp, field);
    auto end = std::chrono::high_resolution_clock::now();
    if (zfpsize == 0) { std::cerr << "Compression failed!" << std::endl; return; }

    double compression_time = std::chrono::duration<double>(end - start).count();
    std::cout << "Compression time: " << compression_time << " s, compressed size = " << zfpsize << " bytes" << std::endl;

    if (processData != 0) {
        std::ofstream outFile1(cpfilename, std::ios::binary);
        outFile1.write(reinterpret_cast<const char*>(buffer), zfpsize);
        outFile1.close();
    }

    T* core_decomp = new T[core_size];
    zfp_field* field_decomp = zfp_field_3d(core_decomp, type, core_x, core_y, core_z);

    stream_rewind(stream);
    start = std::chrono::high_resolution_clock::now();
    zfp_decompress(zfp, field_decomp);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Decompression time: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

    if (!decp_data_host) decp_data_host = new T[full_size];
    for (size_t z = 0; z < core_z; z++) {
        for (size_t y = 0; y < core_y; y++) {
            T* dst = decp_data_host +
                (z + HALO) * width_host * height_host +
                (y + HALO) * width_host +
                HALO;
            const T* src = core_decomp + (z * core_y + y) * core_x;
            std::memcpy(dst, src, core_x * sizeof(T));
        }
    }

    if (processData != 0) {
        std::ofstream outFile(decpfilename, std::ios::binary);
        outFile.write(reinterpret_cast<const char*>(decp_data_host), full_size * sizeof(T));
        outFile.close();
    }


    std::uintmax_t original_dataSize = std::filesystem::file_size(file_path);
    double cr = double(original_dataSize) / double(zfpsize);
    std::cout << "CR (core-only compressed): " << cr << std::endl;

    // --- Cleanup ---
    delete[] core_buffer;
    delete[] core_decomp;
    zfp_field_free(field);
    zfp_field_free(field_decomp);
    zfp_stream_close(zfp);
    stream_close(stream);
    free(buffer);
}

template <typename T>
void compressLocalData_coreOnly(const std::string file_path,
    std::string cpfilename, const std::string filename,
    const T *input_data_host, T *&decp_data_host,
    size_t width_host, size_t height_host, size_t depth_host, // full size with ghost
                                            // ghost thickness
    T &bound, std::string decpfilename, int rank, int processData = 0)
{

    size_t core_x = width_host  - 2 * HALO;
    size_t core_y = height_host - 2 * HALO;
    size_t core_z = depth_host  - 2 * HALO;
    size_t core_size = core_x * core_y * core_z;

    std::cout << "Full block = " << width_host << "×" << height_host << "×" << depth_host
              << ", core = " << core_x << "×" << core_y << "×" << core_z
              << ", core size = " << core_size << std::endl;

    T* core_buffer = new T[core_size];
    for (size_t z = 0; z < core_z; z++) {
        for (size_t y = 0; y < core_y; y++) {
            const T* src = input_data_host +
                (z + HALO) * width_host * height_host +
                (y + HALO) * width_host +
                HALO;
            T* dst = core_buffer + (z * core_y + y) * core_x;
            std::memcpy(dst, src, core_x * sizeof(T));
        }
    }


    auto start = std::chrono::high_resolution_clock::now();
    SZ3::Config conf(static_cast<int>(core_z), static_cast<int>(core_y), static_cast<int>(core_x));
    conf.errorBoundMode = SZ3::EB_ABS;
    conf.absErrorBound = bound;

    size_t cmpSize = 0;
    char *compressedData = SZ_compress(conf, core_buffer, cmpSize);

    auto end = std::chrono::high_resolution_clock::now();
    double compression_time = std::chrono::duration<double>(end - start).count();
    std::cout << "Compression time: " << compression_time
              << " s, cmpSize = " << cmpSize << " bytes" << std::endl;

    if (processData != 0) {
        std::ofstream outFile1(cpfilename, std::ios::binary);
        outFile1.write(compressedData, cmpSize);
        outFile1.close();
    }


    T* core_decomp = new T[core_size];
    start = std::chrono::high_resolution_clock::now();
    SZ_decompress(conf, compressedData, cmpSize, core_decomp);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Decompression time: "
              << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

   
    size_t full_size = width_host * height_host * depth_host;
    if (!decp_data_host) decp_data_host = new T[full_size];
    for (size_t z = 0; z < core_z; z++) {
        for (size_t y = 0; y < core_y; y++) {
            T* dst = decp_data_host +
                (z + HALO) * width_host * height_host +
                (y + HALO) * width_host +
                HALO;
            const T* src = core_decomp + (z * core_y + y) * core_x;
            std::memcpy(dst, src, core_x * sizeof(T));
        }
    }


    if (processData != 0) {
        std::ofstream outFile(decpfilename, std::ios::binary);
        outFile.write(reinterpret_cast<const char*>(decp_data_host), full_size * sizeof(T));
        outFile.close();
    }


    std::uintmax_t original_dataSize = std::filesystem::file_size(file_path);
    double cr = double(original_dataSize) / cmpSize;
    std::cout << "CR (core-only compressed): " << cr << std::endl;

    // === Cleanup ===
    delete[] core_buffer;
    delete[] core_decomp;
    delete[] compressedData;
}

template <typename T>
void compressLocalData(const std::string file_path, std::string cpfilename, const std::string filename,  
    const T *input_data_host, T *&decp_data_host, size_t width_host, size_t height_host, size_t depth_host, 
    T &bound, std::string decpfilename,int rank, int processData = 0) {

    auto start = std::chrono::high_resolution_clock::now();
    SZ3::Config conf(static_cast<int>(depth_host), static_cast<int>(height_host),
    static_cast<int>(width_host));
    
    
    // conf.cmprAlgo = SZ3::ALGO_INTERP_LORENZO;
    conf.errorBoundMode = SZ3::EB_ABS;
    conf.absErrorBound = bound; 

    size_t data_size = width_host * height_host * depth_host;
    std::cout<<"sub datasize: "<< data_size <<std::endl;
    
    char *compressedData = SZ_compress(conf, input_data_host, cmpSize);

    std::cout<<"compression over: "<< cmpSize<<std::endl;

    decp_data_host = new T[data_size];
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    compression_time = double(duration.count())/1000;
    
    start = std::chrono::high_resolution_clock::now();
    SZ_decompress(conf, compressedData, cmpSize, decp_data_host);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout<< "decompression time: "<< duration.count() <<std::endl;
    
    if(processData!=0) {
        std::ofstream outFile1(cpfilename, std::ios::binary);

        if (!outFile1) {
            std::cerr << "Failed to open file for writing compressed data." << std::endl;
        } else {
            outFile1.write(compressedData, cmpSize);
            outFile1.close();
            std::cout << "Compressed data saved to compressed_output.sz" << std::endl;
        }
    }

    delete[] compressedData;

    if(processData != 0){
        std::ofstream outFile(decpfilename, std::ios::binary);
        if (!outFile) {
            std::cerr << "Error: Cannot open output file." << std::endl;
            return;
        }
        outFile.write(reinterpret_cast<const char*>(decp_data_host), data_size * sizeof(T));
        outFile.close();
        std::cout << "Decompressed data saved to decompressed_data.bin" << std::endl;
    }
    
    std::uintmax_t original_dataSize = std::filesystem::file_size(file_path);
    double cr = double(original_dataSize) / cmpSize;
    
    std::cout << "Data read, compressed, and decompressed successfully: "<<cr << std::endl;
}

template <typename T>
void getdata(const std::string &filename, T *&input_data_host, size_t data_size) {

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size != static_cast<std::streamsize>(data_size * sizeof(T))) {
        std::cout<< "datasize is: "<< data_size<< ", datapoint size: "<< sizeof(T)<<std::endl;
        std::cout<<"file size: "<<size<<" calculated size: "<<static_cast<std::streamsize>(data_size * sizeof(T))<<std::endl;
        std::cerr << "File size does not match expected data size." << std::endl;
        return;
    }


    input_data_host = new T[data_size];
    file.read(reinterpret_cast<char *>(input_data_host), size);
    if (!file) {
        std::cerr << "Error reading file." << std::endl;
        return;
    }
    minValue = *std::min_element(input_data_host, input_data_host + data_size);
    maxValue = *std::max_element(input_data_host, input_data_host + data_size);

}

void save_time_records(const std::vector<std::map<std::string, double>>&time_records,
    const std::string& filename) {
   
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }

    for (int i = 0; i < time_records.size(); ++i) {
        for (const auto& it : time_records[i]) {
            ofs << it.first << " " << it.second << "\n";
        }
    }

    ofs.close();

}



__global__ void init_counter(size_t data_size, uint8_t *delta_counter){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=data_size) return;
    delta_counter[i] = 0;
}


template <typename T>
__global__ void extract_false_path(T *decp_data, T *input_data, 
            int *DS_M, int *dec_DS_M, int *AS_M, int *dec_AS_M, 
            size_t data_size, size_t width_host, 
            size_t height_host, size_t depth_host, int *or_types,
            int *de_direction_ds, int *de_direction_as,
            int *edits,uint8_t *delta_counter, int q, T bound, 
            T delta, int rank, int ite,  
            int *send_flag){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= data_size) return;
    int x = i % width_host;
    int y = (i / width_host) % height_host;
    int z = (i / (width_host * height_host)) % depth_host;
    if ((x == 0 || x >= width_host - 1 || y == 0 || y >= height_host - 1 || z == 0 || z >= depth_host - 1 || decp_data[i] == INVALID_VAL)) return;
    
    if(DS_M[i] != dec_DS_M[i]){
        count_f_dir += 1;
        int cur = i;
        
        while (or_types[cur * 2] == de_direction_ds[cur]){
            int next_vertex = or_types[cur * 2];
            if(de_direction_ds[cur]==cur && next_vertex == cur){
                cur = -1;
                break;
            }

            if(next_vertex == cur){
                cur = next_vertex;
                break;
            };
            
            cur = next_vertex;
        }

        int start_vertex = cur;
        
        if (start_vertex!=-1){
            
            int false_index= de_direction_ds[cur];
            int true_index= or_types[cur * 2] == -1?or_types[cur * 2 + 1]:or_types[cur * 2];
            
            if(false_index!=true_index){
                
                if (atomicCAS(&edits[true_index], 0, 1) == 0){
                    
                    applyDeltaBuffer_local<T>(delta_counter,
                                        decp_data, delta, input_data,
                                        bound, q, rank, ite,
                                        width_host, height_host, depth_host, send_flag, true_index);
                }
            }

        }
    }

    if(AS_M[i] != dec_AS_M[i]){
        count_f_dir += 1;
        int cur = i;
        
        while (or_types[cur * 2 + 1] == de_direction_as[cur]){
            int next_vertex = or_types[cur * 2 + 1];
            
            if(de_direction_as[cur]==cur && next_vertex == cur){
                cur = -1;
                break;
            }
            if(next_vertex == cur){
                cur = next_vertex;
                break;
            };
            
            cur = next_vertex;
        }

        int start_vertex = cur;
        
        if (start_vertex!=-1){
            
            int false_index= de_direction_as[cur];
            int true_index= or_types[cur * 2 + 1];
            // if(rank == 2) printf("value at: %lu %d %d %.17f %.17f %.17f %.17f\n", i, false_index, true_index,  decp_data[true_index], input_data[true_index] - bound, decp_data[false_index], input_data[false_index] - bound);
            if(false_index!=true_index){

                if (atomicCAS(&edits[false_index], 0, 1) == 0){
                    
                    // if(rank == 0 && i == 29892) printf("value at: %lu %d %d %.17f %.17f %.17f %.17f %.17f\n", i, false_index, true_index,  decp_data[true_index], input_data[true_index] - bound,decp_data[false_index], input_data[false_index] - bound);
                    applyDeltaBuffer_local<T>(delta_counter,
                                        decp_data, delta, input_data,
                                        bound, q, rank, ite,
                                        width_host, height_host, depth_host, send_flag, false_index);
                }
            }

        }
    }

}


__global__ void count_false_cases(
            int *DS_M, int *dec_DS_M, int *AS_M, int *dec_AS_M, 
            size_t data_size, size_t width_host, 
            size_t height_host, size_t depth_host){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= data_size) return;
    int x = i % width_host;
    int y = (i / width_host) % height_host;
    int z = (i / (width_host * height_host)) % depth_host;
    
    if(AS_M[i] != dec_AS_M[i]){
        count_f_dir += 1;

    }
    
    
    return;

}

template <typename T>
void c_loops(size_t data_size, T *input_data_host, T *decp_data_host, size_t padded_x, 
            size_t padded_y, size_t padded_z, size_t depth_host, int rank, size_t width_host, 
            size_t height_host, T bound, size_t local_x, size_t local_y, size_t local_z, int dims[3], MPI_Comm cart_comm,
            int *DS_M,  int *AS_M, int *de_direction_as, int *de_direction_ds, int *edits, int q, int *or_types, int GhostSize,
            T *sendbuff_right, T *sendbuff_left, T *sendbuff_top, T *sendbuff_bottom, int offset_x, int offset_y, 
            int offset_z, int size, T uTh, T lTh, int iteStep){

    dim3 blockSize(256);
    dim3 gridSize((data_size + blockSize.x - 1) / blockSize.x);
    init_counter<<<gridSize, blockSize>>>(data_size, delta_counter);
    cudaDeviceSynchronize();
    
    int* d_send_flag;
    cudaMalloc(&d_send_flag, sizeof(int));
    int zero = 0;
    cudaMemcpy(d_send_flag, &zero, sizeof(int), cudaMemcpyHostToDevice); 

    int send_flag;
    
    
    iscriticle<T><<<gridSize, blockSize>>>(DS_M, AS_M, de_direction_as, de_direction_ds, all_min, input_data_host, input_data_host, edits, delta_counter, 
        padded_x, padded_y, padded_z, data_size, or_types, offset_x, offset_y, offset_z, width_host, height_host, rank, d_send_flag, q, bound, delta, ite, uTh, lTh, 1);
    cudaError_t err = cudaDeviceSynchronize();
    
    
    if (err != cudaSuccess) {
        printf("Rank %d: iscriticle for ori: %s\n", rank, cudaGetErrorString(err));
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        printf("Rank %d: iscriticle for decp: %s\n", rank, cudaGetErrorString(err));
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    
    MPI_Barrier(cart_comm);
    if (rank == 0) {
        end_time = MPI_Wtime();
        comp_time.push_back(end_time - start_time);
        getfcp += end_time - start_time;
        times.push_back(std::map<std::string, double>{{"comp_time", end_time - start_time}});
        start_time = MPI_Wtime();
    }

    
    int initialValue = 0;
    unsigned int global_count_f_max = 1, global_count_f_min = 1, global_count_f_dir = 0;
    unsigned int host_count_f_min = 1, host_count_f_max = 1, host_count_f_dir = 1;
    
    
    int *un_sign_as;
    cudaMalloc((void**)&un_sign_as, sizeof(int));
    cudaMemset(un_sign_as, 0, sizeof(int));

    int *un_sign_ds;
    cudaMalloc((void**)&un_sign_ds, sizeof(int));
    cudaMemset(un_sign_ds, 0, sizeof(int));
    int h_un_sign_as = data_size,
        h_un_sign_ds = data_size;
    
    std::cout<<"topology preserving started!"<<std::endl;
    fflush(stdout);
    start_time = MPI_Wtime();
    while(global_count_f_max > 0 || global_count_f_min > 0 || global_count_f_dir > 0){
        
        
        ite++;
        
        
        std::vector<float> temp;

        
            
            ite ++ ;
            // if(ite % 10 == 0)  std::cout<< host_count_f_max<<", " << host_count_f_min <<", "<< host_count_f_dir<< ", ite: "<< ite<<", rank: "<<rank << std::endl;
            // if((host_count_f_dir != 0 || host_count_f_max != 0 || host_count_f_min != 0) && rank == 0) std::cout<< host_count_f_max<<", " << host_count_f_min <<", "<< host_count_f_dir<< ", " << ", ite: "<< ite<<", rank: "<<rank << std::endl;
            std::vector<float> c_temp;
            init_edits<<<gridSize, blockSize>>>(data_size, edits);
            
            err = cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(unsigned int));
            err = cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(unsigned int));
            err = cudaMemcpyToSymbol(count_f_dir, &initialValue, sizeof(unsigned int));
            if (err != cudaSuccess) {
                printf("Rank %d: copy counter %d: %s\n", rank, ite, cudaGetErrorString(err));
                fflush(stdout);
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
            cudaDeviceSynchronize();

            iscriticle<T><<<gridSize, blockSize>>>(dec_DS_M, dec_AS_M, de_direction_as, de_direction_ds, all_min, input_data_host, decp_data_host, edits, delta_counter, 
            padded_x, padded_y, padded_z, data_size, or_types, offset_x, offset_y, offset_z, width_host, height_host, rank, d_send_flag, q, bound, delta, ite, uTh, lTh);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("Rank %d: error is criticle %d: %s\n", rank, ite, cudaGetErrorString(err));
                fflush(stdout);
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
            
            cudaMemcpy(&send_flag, d_send_flag, sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            
           
            cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&host_count_f_dir, count_f_dir, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
            
            
            cudaDeviceSynchronize();
           
        temp.push_back(MPI_Wtime() - start_time);
       
        exchange_ghost_layers_cuda_aware<uint8_t>(decp_data_host, delta_counter, input_data_host, bound,  edits, padded_x, padded_y, padded_z, width_host, height_host, depth_host, local_x, local_y, local_z, offset_x, offset_y, offset_z, delta, cart_comm, dims, rank, send_flag, ite);
        
        err = cudaMemcpy(d_send_flag, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
       
        
        temp.push_back(MPI_Wtime() - start_time);

        err = cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(unsigned int));
        err = cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(unsigned int));
        err = cudaMemcpyToSymbol(count_f_dir, &initialValue, sizeof(unsigned int));
        if (err != cudaSuccess) {
            printf("Rank %d: copy counter %d: %s\n", rank, ite, cudaGetErrorString(err));
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        cudaDeviceSynchronize();

        iscriticle<T><<<gridSize, blockSize>>>(dec_DS_M, dec_AS_M, de_direction_as, de_direction_ds, all_min, input_data_host, decp_data_host, edits, delta_counter, 
        padded_x, padded_y, padded_z, data_size, or_types, offset_x, offset_y, offset_z, width_host, height_host, rank, d_send_flag, q, bound, delta, ite, uTh, lTh, 0, 0, 1);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Rank %d: error is criticle %d: %s\n", rank, ite, cudaGetErrorString(err));
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        
        cudaMemcpy(&send_flag, d_send_flag, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        

        cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&host_count_f_dir, count_f_dir, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
        
        
        cudaDeviceSynchronize();
        
        end_time = MPI_Wtime();
        temp.push_back(end_time - start_time);
        
        MPI_Allreduce(&host_count_f_max, &global_count_f_max, 1, MPI_UNSIGNED, MPI_MAX, cart_comm);
        MPI_Allreduce(&host_count_f_min, &global_count_f_min, 1, MPI_UNSIGNED, MPI_MAX, cart_comm);
        MPI_Allreduce(&host_count_f_dir, &global_count_f_dir, 1, MPI_UNSIGNED, MPI_MAX, cart_comm);

       
        end_time = MPI_Wtime();
        temp.push_back(end_time - start_time);
        
        time_counter.push_back(temp);
    }

    
    if(rank == 0) printf("iteration is: %d\n", ite);
}


std::string extractFilename(const std::string& path) {
    
    int lastSlash = path.find_last_of("/\\");
    std::string filename = (lastSlash == std::string::npos) ? path : path.substr(lastSlash + 1);

    int dotPos = filename.find_last_of('.');
    std::string name = (dotPos == std::string::npos) ? filename : filename.substr(0, dotPos);

    return name;
}



template <typename T>
__global__ void extract_edits_kernel(
    const T*  decp_data,
    const T*  decp_data_copy,
    const T*  input_data,
    const uint8_t*  delta_counter,
    uint64_t* diffs_out,
    uint8_t* deltas_out,
    T* edits_out,
    size_t local_x, size_t local_y, size_t local_z,
    size_t padded_x, size_t padded_y, 
    int q, T bound, size_t data_size, unsigned int* d_edit_count, T delta, int rank
) {
    
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= data_size) return;
    
    int x = i % padded_x;
    int y = (i / padded_x) % padded_y;
    int z = (i / (padded_x * padded_y)) % (local_z + 2);
    
    size_t localId = (x - HALO) + (y - HALO) * local_x + (z - HALO) * local_x * local_y;
    if(x == 0 || x == padded_x - 1 || y == 0 || y == padded_y - 1 || z == 0 || z == local_z + 1) return;
    T currentV = decp_data[i];
    T previousV = decp_data_copy[i];
    
    if (currentV != previousV) {
        // if(rank == 0) printf("2 here : %d %d %d %d %d %d\n", x, y, z, padded_x, padded_y, padded_z);
        if(currentV < previousV){
            unsigned int idx = atomicAdd(d_edit_count, 1);
            // diffs_out[idx] = i;
            if(currentV != input_data[i] - bound) {
                deltas_out[localId] = static_cast<uint8_t>(std::round((previousV - currentV) / delta));

                edits_out[localId] = 0;

            }
            else{
                deltas_out[localId] = q;
                edits_out[localId] = input_data[i] - bound;
            }
            
        }
    }
    else{
        deltas_out[localId] = 0;
        edits_out[localId] = 0;
    }
}

template <typename T>
void run_extract_edits(
    const T* d_decp_data,
    const T* d_decp_data_copy,
    const T* d_input_data,
    const uint8_t* d_delta_counter,
    size_t local_x, size_t local_y, size_t local_z,
    size_t padded_x, size_t padded_y,
    int q, T bound, 
    std::vector<uint64_t>& h_diffs,
    std::vector<uint8_t>& h_deltas,
    std::vector<T>& h_edits, std::vector<uint64_t>& h_edit_pos, unsigned int &h_count, int rank
) {
    uint64_t* d_diffs;
    uint8_t* d_deltas;
    T* d_edits;
    
    size_t data_size = local_x * local_y * local_z;
    size_t num_Elements = padded_x * padded_y * (local_z + 2);
    
    cudaError_t err = cudaMalloc(&d_deltas, data_size * sizeof(uint8_t));
    if (err != cudaSuccess) {
        printf("Rank cudaMalloc failed Malloc for deltas: %s\n", cudaGetErrorString(err));
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }


    err = cudaMalloc(&d_edits, data_size * sizeof(T));
    if (err != cudaSuccess) {
        printf("Rank : cudaMalloc failed Malloc for edits: %s\n", cudaGetErrorString(err));
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, -1);
        // MPI_Abort(MPI_COMM_WORLD, -1);
    }
    cudaMemset(d_edits, 0, data_size * sizeof(T)); 

   
    
    unsigned int* d_edit_count;
    cudaMalloc(&d_edit_count, sizeof(unsigned int));
    cudaMemset(d_edit_count, 0, sizeof(unsigned int));

    dim3 blockSize(256);
    dim3 gridSize((num_Elements + blockSize.x - 1) / blockSize.x);
    
    extract_edits_kernel<T><<<gridSize, blockSize>>>(
        d_decp_data,
        d_decp_data_copy,
        d_input_data,
        d_delta_counter,
        d_diffs,
        d_deltas,
        d_edits,
        local_x, local_y, local_z,
        padded_x, padded_y,
        q, bound, num_Elements, d_edit_count, delta, rank
    );
    
    cudaDeviceSynchronize();
    
    
    cudaMemcpy(&h_count, d_edit_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf("Edit count: %u\n", h_count);

    thrust::device_vector<T> d_edit_val_compact;
    compact_edits_dense_to_sparse<T>(d_edits, data_size, d_edit_val_compact);
    
    h_deltas.resize(data_size);
    h_edits.resize(d_edit_val_compact.size()); 
    
    cudaMemcpy(h_deltas.data(), d_deltas, data_size* sizeof(uint8_t), cudaMemcpyDeviceToHost);
    thrust::copy(d_edit_val_compact.begin(), d_edit_val_compact.end(), h_edits.begin());
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Rank : thrust failed Malloc for edits: %s\n", cudaGetErrorString(err));
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    
    cudaFree(d_deltas);
    cudaFree(d_edits);

}



template <typename T>
std::vector<uint8_t> extract_samples(const T* data, size_t num_elems, size_t max_bytes = 65536) {
    std::vector<uint8_t> result;
    size_t elem_size = sizeof(T);
    size_t max_elems = max_bytes / elem_size;
    size_t step = std::max((size_t)1, num_elems / max_elems);
    for (size_t i = 0; i < num_elems && result.size() + elem_size <= max_bytes; i += step) {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&data[i]);
        result.insert(result.end(), ptr, ptr + elem_size);
    }
    return result;
}


template <typename T>
MPI_Datatype get_mpi_datatype() {
    if constexpr (std::is_same<T, int>::value) return MPI_INT;
    else if constexpr (std::is_same<T, uint8_t>::value) return MPI_UINT8_T;
    else if constexpr (std::is_same<T, uint16_t>::value) return MPI_UINT16_T;
    else if constexpr (std::is_same<T, uint32_t>::value) return MPI_UINT32_T;
    else if constexpr (std::is_same<T, float>::value) return MPI_FLOAT;
    else if constexpr (std::is_same<T, double>::value) return MPI_DOUBLE;
    else {
        std::cerr << "Unsupported type for MPI I/O.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}




template <typename T>
std::vector<uint8_t> compress_zstd(const std::vector<T>& data) {
    if (data.empty()) return {};

    const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(data.data());
    size_t src_size = data.size() * sizeof(T);
    size_t bound = ZSTD_compressBound(src_size);

    std::vector<uint8_t> compressed(bound);
    size_t csize = ZSTD_compress(compressed.data(), bound, src_ptr, src_size, 3);
    if (ZSTD_isError(csize)) {
        std::cerr << "Compression failed: " << ZSTD_getErrorName(csize) << "\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    compressed.resize(csize);
    return compressed;
}




inline std::vector<uint8_t> zstd_compress(const uint8_t* data, size_t n, int level = 5) {
    size_t bound = ZSTD_compressBound(n);
    std::vector<uint8_t> out(bound);
    size_t csz = ZSTD_compress(out.data(), bound, data, n, level);
    if (ZSTD_isError(csz)) throw std::runtime_error(ZSTD_getErrorName(csz));
    out.resize(csz);
    return out;
}


std::vector<uint8_t> zstd_compress(const void* src, size_t src_sz, int level);
std::vector<uint8_t> zstd_decompress(const void* comp, size_t csz, size_t expect_sz);


template <typename T>
std::vector<T> decompress_zstd(const std::vector<uint8_t>& comp) {
    if (comp.size() < sizeof(size_t)) {
        throw std::runtime_error("Compressed data too small to contain expected size.");
    }

    size_t num_elements;
    std::memcpy(&num_elements, comp.data(), sizeof(size_t));

    const uint8_t* comp_data = comp.data() + sizeof(size_t);
    size_t comp_size = comp.size() - sizeof(size_t);

    size_t expected_bytes = num_elements * sizeof(T);
    std::vector<T> out(num_elements);

    size_t actual_size = ZSTD_decompress(out.data(), expected_bytes, comp_data, comp_size);
    if (ZSTD_isError(actual_size)) throw std::runtime_error(ZSTD_getErrorName(actual_size));
    if (actual_size != expected_bytes) throw std::runtime_error("zstd decompress: size mismatch");
    return out;
}

inline std::vector<uint8_t> zstd_decompress(const uint8_t* comp, size_t csz, size_t expected_bytes) {
    std::vector<uint8_t> out(expected_bytes);
    size_t n = ZSTD_decompress(out.data(), expected_bytes, comp, csz);
    if (ZSTD_isError(n)) throw std::runtime_error(ZSTD_getErrorName(n));
    if (n != expected_bytes) throw std::runtime_error("zstd: size mismatch");
    return out;
}

inline std::vector<uint8_t>
pack_nbits(const uint8_t* symbols, size_t n_symbols, unsigned nbits) {
    if (nbits == 0 || nbits > 16) throw std::invalid_argument("nbits must be 1..16");
    std::vector<uint8_t> out;
    out.reserve((n_symbols * nbits + 7) / 8);

    uint64_t bitbuf = 0; unsigned bitcnt = 0;
    const uint64_t mask = (1ULL << nbits) - 1ULL;
    for (size_t i = 0; i < n_symbols; ++i) {
        const uint64_t v = symbols[i];
        if (v > mask) throw std::out_of_range("symbol exceeds 0..N range");
        bitbuf |= (v & mask) << bitcnt;
        bitcnt += nbits;
        while (bitcnt >= 8) {
            out.push_back(static_cast<uint8_t>(bitbuf & 0xFF));
            bitbuf >>= 8; bitcnt -= 8;
        }
    }
    if (bitcnt) out.push_back(static_cast<uint8_t>(bitbuf & 0xFF));
    return out;
}



inline unsigned bits_for_max(unsigned N) {

    if (N == 0) return 1;
    return (unsigned)std::ceil(std::log2((double)N + 1.0));
}


inline std::vector<uint8_t>
compress_symbols_0_N(const std::vector<uint8_t>& symbols,
                     unsigned N,               
                     int level = 5,
                     size_t* packed_bytes_out = nullptr)
{
    const unsigned nbits = bits_for_max(N);
    auto packed = pack_nbits(symbols.data(), symbols.size(), nbits);
    if (packed_bytes_out) *packed_bytes_out = packed.size();
    return zstd_compress(packed.data(), packed.size(), level);
}

template <typename T>
std::vector<T> decompress_zstd_to_vector(const std::vector<uint8_t>& compressed, size_t expected_size) {
    std::vector<T> output(expected_size);

    size_t decompressed_bytes = ZSTD_decompress(
        output.data(), expected_size * sizeof(T),
        compressed.data(), compressed.size()
    );

    if (ZSTD_isError(decompressed_bytes)) {
        std::cerr << "ZSTD decompression error: " << ZSTD_getErrorName(decompressed_bytes) << "\n";
        std::exit(1);
    }

    if (decompressed_bytes != expected_size * sizeof(T)) {
        std::cout << "Warning: decompressed size mismatch: " << expected_size * sizeof(T) << ", "<< decompressed_bytes<<std::endl;
    }

    return output;
}

inline std::vector<uint8_t>
unpack_nbits(const uint8_t* packed, size_t packed_bytes, size_t n_symbols, unsigned nbits) {
    if (nbits == 0 || nbits > 16) throw std::invalid_argument("nbits must be 1..16");
    std::vector<uint8_t> out(n_symbols);
    uint64_t bitbuf = 0; unsigned bitcnt = 0; size_t in_pos = 0;
    const uint64_t mask = (1ULL << nbits) - 1ULL;

    for (size_t i = 0; i < n_symbols; ++i) {
        while (bitcnt < nbits) {
            if (in_pos >= packed_bytes) throw std::runtime_error("packed too small");
            bitbuf |= (uint64_t)packed[in_pos++] << bitcnt;
            bitcnt += 8;
        }
        out[i] = static_cast<uint8_t>(bitbuf & mask);
        bitbuf >>= nbits; bitcnt -= nbits;
    }
    return out;
}

inline size_t packed_size_bytes(size_t n_symbols, unsigned nbits) {
    return (n_symbols * nbits + 7) / 8;
}


inline std::vector<uint8_t>
decompress_symbols_0_N(const uint8_t* comp,
                       size_t csz,
                       size_t n_symbols,
                       unsigned N)            
{
    const unsigned nbits = bits_for_max(N);
    const size_t need = packed_size_bytes(n_symbols, nbits);
    auto packed = zstd_decompress(comp, csz, need);
    return unpack_nbits(packed.data(), packed.size(), n_symbols, nbits);
}


template <typename T>
void gather_cost_sparse(std::string file_path, std::string filename, T* decp_data, T* decp_data_host, T* decp_data_copy_host, T* decp_data_copy,
                  T* input_data, T* decp_data_all, T* decp_data_copy_all, std::string compressor_id, uint8_t* delta_counter,
                  int q, T bound, size_t num_Elements, size_t local_x, size_t local_y, size_t local_z,
                  size_t padded_x, size_t padded_y, size_t padded_z, size_t width_host, size_t height_host, size_t depth_host, 
                  int rank, int world_size, T delta,
                  MPI_Comm comm, unsigned int &h_count, std::vector<double>& edits_time,
                  int PX, int PY, int PZ
                ) {

    using index_t = uint64_t;
    size_t data_size = local_x * local_y * local_z;
    size_t rankSize = padded_x * padded_y * padded_z;
    std::vector<index_t> diffs, edit_pos;
    std::vector<T> edits;

    std::vector<uint8_t> deltas, deltas1;

    
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();

    run_extract_edits<T>(decp_data, decp_data_copy, input_data, delta_counter,
                   local_x, local_y, local_z, padded_x, padded_y,
                   q, bound, diffs, deltas, edits, edit_pos, h_count, rank);
    
    
    
    MPI_Barrier(comm);
    
    unsigned int lcnt = edits.size();
    std::vector<unsigned> h_counts(world_size);
    MPI_Gather(&lcnt, 1, MPI_UNSIGNED, h_counts.data(), 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);


    double t1 = MPI_Wtime();
    if (rank == 0) std::cout << "[Time] extract_edits: " << t1 - t0 << " s\n";
    edits_time.push_back(t1-t0);


    MPI_Barrier(comm);
    double t2 = MPI_Wtime();
    if (rank == 0) std::cout << "[Time] encode_diffs: " << t2 - t1 << " s"<< "with size: "<< diffs.size()<<"\n";
    
    std::vector<uint8_t> comp_diffs  = {};
    std::vector<uint8_t> comp_edits  = compress_zstd<T>(edits);
    std::vector<uint8_t> comp_deltas = compress_zstd<uint8_t>(deltas);
    std::vector<uint8_t> comp_edit_pos = {};
    
    std::cout << "compressed deltas: "<< comp_deltas.size()<< ", " << comp_edits.size() << ", "<< comp_edit_pos.size() <<std::endl;
    size_t packed_bytes = 0;

    comp_deltas = compress_symbols_0_N(deltas, /*N=*/q, /*level=*/5);
    std::cout << "packed_bytes=" << packed_bytes
            << ", compressed=" << comp_deltas.size() << " bytes\n";


    double t3 = MPI_Wtime();
    
    size_t sz_diff = comp_diffs.size(), sz_edit = comp_edits.size(), sz_edit_pos = comp_edit_pos.size(), sz_delta = comp_deltas.size();
    if (rank == 0) std::cout << "[Time] compress: " << t3 - t2 <<", "<< sz_diff <<"/" << diffs.size() * 8 << " s\n";
    edits_time.push_back(t3-t2);
    std::vector<size_t> all_sz_diff(world_size), all_sz_edit(world_size), all_sz_edit_pos(world_size), all_sz_delta(world_size);
    
    MPI_Allgather(&sz_delta,1, MPI_UINT64_T, all_sz_delta.data(),1, MPI_UINT64_T, comm);
    MPI_Allgather(&sz_edit, 1, MPI_UINT64_T, all_sz_edit.data(), 1, MPI_UINT64_T, comm);
    
    storageOverhead = sz_diff + sz_edit + sz_delta + sz_edit_pos;
    
    std::vector<size_t> off_diff(world_size, 0), off_edit(world_size, 0), off_delta(world_size, 0), off_edit_pos(world_size, 0);
    
    for (int i = 1; i < world_size; ++i) {
        
        off_edit[i]  = off_edit[i-1]  + all_sz_edit[i-1];
        
        off_delta[i] = off_delta[i-1] + all_sz_delta[i-1];

    }
    
    MPI_Barrier(comm);
    double t4 = MPI_Wtime();
    if (rank == 0) std::cout << "[Time] offset size: " << t4 - t3 << " s\n";
    
    MPI_File f_diff, f_edit, f_delta, f_edit_pos;
    std::string prefix = "";
    
    
    MPI_File_open(comm, (prefix + "_edits.bin").c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f_edit);
    MPI_File_open(comm, (prefix + "_deltas.bin").c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f_delta);
   
    MPI_File_write_at_all(f_edit,  off_edit[rank],  comp_edits.data(),  sz_edit,  MPI_BYTE, MPI_STATUS_IGNORE);
    MPI_File_write_at_all(f_delta, off_delta[rank], comp_deltas.data(), sz_delta, MPI_BYTE, MPI_STATUS_IGNORE);
    
    MPI_File_close(&f_edit);
    MPI_File_close(&f_delta);
    
   
    size_t core_data_size = local_x * local_y * local_z;

   
    std::vector<size_t> all_core_sizes;
    if (rank == 0) all_core_sizes.resize(world_size);

    MPI_Gather(&core_data_size, 1, MPI_UINT64_T,
            all_core_sizes.data(), 1, MPI_UINT64_T,
            0, MPI_COMM_WORLD);


    MPI_Barrier(comm);
    double t5 = MPI_Wtime();
    if (rank == 0) std::cout << "[Time] MPI write: " << t5 - t4 << " s\n";
    edits_time.push_back(t5-t4);
    
    return;

    
}

template <typename T>
void validateResult(T *decp_data, T*decp_data_copy, size_t numElements, size_t expected_size) {

    std::string prefix = "";

    
    MPI_File f_edit;
    MPI_File_open(MPI_COMM_SELF, (prefix + "_edits.bin").c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &f_edit);

    MPI_Offset file_size_edit = 0;
    MPI_File_get_size(f_edit, &file_size_edit);

    std::vector<uint8_t> comp_edits(file_size_edit);
    MPI_File_read_at_all(f_edit, 0, comp_edits.data(), file_size_edit, MPI_BYTE, MPI_STATUS_IGNORE);
    MPI_File_close(&f_edit);

    
    MPI_File f_delta;
    MPI_File_open(MPI_COMM_SELF, (prefix + "_deltas.bin").c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &f_delta);

    MPI_Offset file_size_delta = 0;
    MPI_File_get_size(f_delta, &file_size_delta);

    std::vector<uint8_t> comp_deltas(file_size_delta);
    MPI_File_read_at_all(f_delta, 0, comp_deltas.data(), file_size_delta, MPI_BYTE, MPI_STATUS_IGNORE);
    MPI_File_close(&f_delta);

    std::cout << "[Rank 0] Loaded compressed files. Sizes: " 
              << comp_edits.size() << " (edits), " 
              << comp_deltas.size() << " (deltas)" << std::endl;


    std::vector<T> edits = decompress_zstd_to_vector<T>(comp_edits, expected_size);
    std::cout << "[Rank 0] Decompressed sizes: " << edits.size() << std::endl;
    
    std::vector<uint8_t> deltas = decompress_symbols_0_N(comp_deltas.data(), file_size_delta, numElements, /*N=*/q  /*expected_size=*/);
    
    std::cout << "[Rank 0] Decompressed sizes: " << edits.size() << " edits, " << deltas.size() << " deltas" << std::endl;
    return;
}


inline bool rank_has_z(size_t target_z, size_t offset_z, size_t local_z) {
    return (target_z >= offset_z) && (target_z < offset_z + local_z); 
}


__global__ void warmup_kernel(double* data, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0 + 1.0;
    }
}


int main(int argc, char** argv) {
    
    
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);

    
    int local_rank = -1;
    if (getenv("SLURM_LOCALID"))
        local_rank = atoi(getenv("SLURM_LOCALID"));
    else if (getenv("OMPI_COMM_WORLD_LOCAL_RANK"))
        local_rank = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
    else if (getenv("MV2_COMM_WORLD_LOCAL_RANK"))
        local_rank = atoi(getenv("MV2_COMM_WORLD_LOCAL_RANK"));

    if (local_rank >= 0) {
        int dev = local_rank % dev_count;
        cudaSetDevice(dev);
        printf("Rank %d (local %d) using GPU %d\n", rank, local_rank, dev);
    }

    
    std::cout << std::fixed << std::setprecision(16);
    std::string dimension = argv[1];
    
    
    double bound = std::stod(argv[2]);
    std::string compressor_id = argv[3];
    int q = 5;
    PX = std::stoi(argv[4]);
    PY = std::stoi(argv[5]);
    PZ = std::stoi(argv[6]); 
   
    check = 0; 
   
    double uTh = 0.0; 
    double lTh = 0.0;
    
    int iteStep = 0;
    int compression = 1;
    int check_consistency = 0;
    int compression_mode = 1;
    int weakscaling = 0;

    size_t width_host, height_host, depth_host;
    std::string file_path;
    std::istringstream iss(dimension);
    char delimiter;

    
    
    if (std::getline(iss, file_path, ',')) {
        if (iss >> width_host >> delimiter && delimiter == ',' &&
            iss >> height_host >> delimiter && delimiter == ',' &&
            iss >> depth_host) {
                
        } else {
            std::cerr << "Parsing error for dimensions" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else {
        std::cerr << "Parsing error for file" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    std::string filename = extractFilename(file_path);
    size_t num_Elements = static_cast<int>(width_host) * height_host * depth_host;
    size_t global_x = width_host, global_y = height_host, global_z = depth_host;
    int px = PX, py = PY, pz = PZ;
    int dims[3] = {px, py, pz};
    int periods[3] = {0, 0, 0};
    
    
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);

    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);
    int coords_x = coords[0], coords_y = coords[1], coords_z = coords[2];

    if (px * py * pz != size) {
        if (rank == 0)
            std::cerr << "ERROR: PX * PY = " << (px * py) << " != total ranks = " << size << "\n";
       
    }
    
    // X dimension
    int base_local_x = width_host / px;
    int remainder_x = width_host % px;
    size_t local_x = base_local_x + (coords_x < remainder_x ? 1 : 0);
    size_t offset_x =static_cast<int>(coords_x) * base_local_x + std::min(coords_x, remainder_x);
    
    // Y dimension
    int base_local_y = height_host / py;
    int remainder_y = height_host % py;
    size_t local_y = base_local_y + (coords_y < remainder_y ? 1 : 0);
    size_t offset_y = static_cast<int>(coords_y) * base_local_y + std::min(coords_y, remainder_y);
    
    // Z dimension
    int base_local_z = depth_host / pz;
    int remainder_z = depth_host % pz;
    size_t local_z = base_local_z + (coords_z < remainder_z ? 1 : 0);
    size_t offset_z = static_cast<int>(coords_z) * base_local_z + std::min(coords_z, remainder_z);


    size_t padded_x = local_x + 2 * HALO;
    size_t padded_y = local_y + 2 * HALO;
    size_t padded_z = local_z + 2 * HALO;

    
    int world_rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int cart_rank = -1;
    if (cart_comm != MPI_COMM_NULL) MPI_Comm_rank(cart_comm, &cart_rank);

    int N = 1 << 20;  // 1M doubles
    double* d_data = nullptr;
    cudaMalloc(&d_data, N * sizeof(double));
    cudaMemset(d_data, 0, N * sizeof(double));

    // 1. GPU kernel warm-up
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    for (int i = 0; i < 10; i++) {
        warmup_kernel<<<grid, block>>>(d_data, N);
        cudaDeviceSynchronize();
    }

    // 2. MPI warm-up
    for (int i = 0; i < 10; i++) {
        double local_sum = rank + i;
        double global_sum = 0.0;
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        std::cout << "Warm-up done" << std::endl;
    }

    cudaFree(d_data);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "Starting benchmark..." << std::endl;
   
    size_t data_size = static_cast<int>(padded_x) * padded_y * padded_z;
    
   
    double end2end = 0.0;
    MPI_Barrier(cart_comm);
    if(rank == 0) end2end = MPI_Wtime();

    double* d_oriData = nullptr;
    double* d_decData = nullptr;
    unsigned char* d_cmpBytes = nullptr;
    double* decompressed_host_data = nullptr;
    double *input_data = nullptr;
    double *decp_data = nullptr;
    double* input_data_host = new double[data_size];
    double* decp_data_host = new double[data_size];
    double* decp_data_copy_host = new double[data_size];
    
    
    std::string decpfilename = "decp_"+filename+"_"+compressor_id+'_'+std::to_string(bound)+".bin";
    std::string cpfilename;
    

    int result = 0;
    std::string command;
    if(rank == 0 && compression == 1 && compression_mode == 0){
        double range  = bound;
        
        input_data = new double[num_Elements];
        getdata<double>(file_path, input_data, num_Elements);        
        std::ostringstream oss1;
        oss1 << std::scientific << std::setprecision(17) << bound;
       
        cpfilename = "compressed_"+filename+"_"+std::to_string(bound)+".sz";
        if(compressor_id == "sz3") {
            
            compressLocalData<double>(file_path, cpfilename, filename, input_data, decp_data, width_host, height_host, depth_host, bound, decpfilename, rank, 1);

        }
        else if(compressor_id=="zfp"){
            auto start = std::chrono::high_resolution_clock::now();
            cpfilename = "compressed_"+filename+"_"+std::to_string(bound)+".zfp";
            
            compressLocalDataZFP<double>(file_path, cpfilename, filename, input_data_host, decp_data_host, padded_x, padded_y, padded_z, bound, decpfilename, rank, 1);
           
        }

    }

    if(compressor_id == "sz3") cpfilename = "compressed_"+filename+"_"+std::to_string(bound)+".sz";
    else if(compressor_id == "zfp") cpfilename = cpfilename = "compressed_"+filename+"_"+std::to_string(bound)+".zfp";
    

    MPI_Bcast(&bound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    std::cout << "Rank " << rank << " got value = " << bound << std::endl;
    std::cout<<decpfilename<<std::endl;
    

    if(rank == 0){
        cudaFree(d_oriData);
        cudaFree(d_decData);
        cudaFree(d_cmpBytes);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    delta = bound / (q);
    double dataStart = 0.0;
    double dataTime = 0.0;
    MPI_Barrier(cart_comm);
    if(rank == 0) dataStart = MPI_Wtime();


    read_subblock<double>(file_path.c_str(),
                   offset_x, offset_y, offset_z,
                   local_x, local_y, local_z,
                   global_x, global_y, global_z,
                   input_data_host);
    MPI_Barrier(cart_comm);
    
    exchange_ghost_layers<double>(input_data_host, padded_x, padded_y, padded_z, local_x, local_y, local_z, cart_comm, dims, rank);
    
    if(compression_mode == 1 && compressor_id == "sz3") compressLocalData_coreOnly<double>(file_path, cpfilename, filename, input_data_host, decp_data_host, padded_x, padded_y, padded_z, bound, decpfilename, rank);
    else if(compression_mode == 1 && compressor_id == "zfp") compressLocalDataZFP_coreOnly<double>(file_path, cpfilename, filename, input_data_host, decp_data_host, padded_x, padded_y, padded_z, local_x, local_y, local_z, bound, decpfilename, rank);

    cudaError_t err = cudaMalloc(&de_direction_as, data_size * sizeof(int));
    err = cudaMalloc(&de_direction_ds, data_size * sizeof(int));

    if (err != cudaSuccess) {
        printf("Rank %d: cudaMalloc failed Malloc for all_max: %s\n", rank, cudaGetErrorString(err));
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    cudaMalloc(&delta_counter, data_size * sizeof(uint8_t));
    
    cudaMalloc(&edits, data_size * sizeof(int));
    cudaMemset(edits, 0, data_size * sizeof(int));  
    
    err = cudaMalloc(&or_types, 2*data_size * sizeof(int));
    if (err != cudaSuccess) {
        printf("Rank %d: cudaMalloc failed Malloc for or_types: %s\n", rank, cudaGetErrorString(err));
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    
    double *device_input, *device_decp, *device_decp_copy;
    size_t local_bytes = data_size * sizeof(double);
    err = cudaMalloc((void**)&device_input, local_bytes);
    if (err != cudaSuccess) {
        printf("Rank %d: cudaMalloc failed Malloc for device_input: %s\n", rank, cudaGetErrorString(err));
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    cudaMalloc((void**)&device_decp, local_bytes);
    cudaMalloc((void**)&device_decp_copy, local_bytes);
    
    if(compression_mode == 0) read_subblock<double>(decpfilename.c_str(), offset_x, offset_y, offset_z, local_x, local_y, local_z, global_x, global_y, global_z, decp_data_host);
    MPI_Barrier(cart_comm);
    exchange_ghost_layers<double>(decp_data_host, padded_x, padded_y, padded_z, local_x, local_y, local_z, cart_comm, dims, rank);
    err = cudaMemcpy(device_input, input_data_host, local_bytes, cudaMemcpyHostToDevice);

    
    if (err != cudaSuccess) {
        printf("Rank %d: compression failed: %s\n", rank, cudaGetErrorString(err));
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }



    std::memcpy(decp_data_copy_host, decp_data_host, data_size * sizeof(double));
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Rank %d: before compression failed: %s\n", rank, cudaGetErrorString(err));
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    
    
    err = cudaMemcpy(device_decp, decp_data_host, local_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Rank %d: cudaMalloc failed Malloc for device_decp: %s\n", rank, cudaGetErrorString(err));
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    
    
    cudaMemcpy(device_decp_copy, decp_data_host, local_bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time_t = 0.0;
    double dataEnd = 0.0;
    if(rank == 0){
        dataEnd = MPI_Wtime();
        dataTime =  dataEnd - dataStart;
        start_time_t = MPI_Wtime();
    }

    dim3 blockSize(256);
    dim3 gridSize((data_size + blockSize.x - 1) / blockSize.x);

    c_loops<double>(data_size, device_input, device_decp, padded_x, 
        padded_y, padded_z, depth_host, rank, width_host, 
        height_host, bound, local_x, local_y, local_z, dims, cart_comm, DS_M, AS_M, de_direction_as, de_direction_ds, edits, q, or_types, 0,
        sendbuff_right, sendbuff_left, sendbuff_top, sendbuff_bottom, offset_x, offset_y, offset_z, size, uTh, lTh, iteStep
    );
    
    MPI_Barrier(cart_comm);
    int c1 = 0;  
    std::ofstream outFilep("Global_performance_with_ite_cuda_"+compressor_id+"_"+filename+"_rank_"+std::to_string(rank)+"global_.txt", std::ios::app);
       
    if (!outFilep) {
        std::cerr << "Unable to open file for writing." << std::endl;
        return 1; 
    }

    outFilep << "iteStep: "<<iteStep <<std::endl;
    for (const auto& row : time_counter) {
        outFilep << "version 2 iteration: "<<c1<<": ";
        for (size_t i = 0; i < row.size(); ++i) {
            outFilep << row[i];
            if (i != row.size() - 1) { 
                outFilep << ", ";
            }
        }
        
        outFilep << std::endl;
        c1+=1;
    }
    outFilep << "\n"<< std::endl;

    
    
    cudaDeviceSynchronize();

    cudaFree(all_max);
    cudaFree(DS_M);
    cudaFree(AS_M);
    cudaFree(de_direction_as);
    cudaFree(de_direction_ds);
    cudaFree(sendbuff_right);
    cudaFree(sendbuff_left);
    cudaFree(sendbuff_top);
    cudaFree(sendbuff_bottom);
   
    
    unsigned int h_count = 0;
    std::vector<double> edits_time = {};
    
    cudaMemcpy(decp_data_host, device_decp, local_bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::vector<double> core(local_x * local_y * local_z);
    std::vector<double> global, global_copy;

        


    if(weakscaling == 0) {
        gather_cost_sparse<double>(file_path, filename, device_decp, decp_data_host, decp_data_copy_host, device_decp_copy,
          device_input, global.data(), global_copy.data(), compressor_id, delta_counter,
          q, bound, num_Elements, local_x, local_y, local_z,
          padded_x, padded_y, padded_z, width_host, height_host, depth_host, rank, size, delta,
          cart_comm, h_count, edits_time, PX, PY, PZ);
        
    }
   
    
    MPI_Barrier(cart_comm);
    double WholeTime;
    if(rank == 0) {
        double end2end1 = MPI_Wtime();
        WholeTime = end2end1 - end2end;
        editsTime = end2end1 - start_time;
        std::cout<<"whole time: "<< WholeTime <<"s "<<std::endl;

    }


    

    MPI_Barrier(cart_comm);


    
    std::uintmax_t total_cmpsize = 0;
    MPI_Reduce(&cmpSize, &total_cmpsize, 1, MPI_UNSIGNED_SHORT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    std::uintmax_t total_storageOverhead = 0;
    MPI_Reduce(&storageOverhead, &total_storageOverhead, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(cart_comm);
        
    unsigned int total_edited = 0;
    MPI_Reduce(&h_count, &total_edited, 1, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(cart_comm);

    int global_max;
    MPI_Reduce(&ite, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

        
    decp_data = new double[num_Elements];

    
    cudaFree(device_input);
    cudaFree(device_decp);
    cudaFree(dec_AS_M);
    cudaFree(dec_DS_M);
   
    MPI_Barrier(cart_comm);
    


    
    if(check_consistency == 1 && rank == 0){
        double *device_input_all, *device_decp_all;
        dim3 blockSize(256);
        dim3 gridSize((num_Elements + blockSize.x - 1) / blockSize.x);
        

        err = cudaMalloc(&DS_M, num_Elements * sizeof(int));
        err = cudaMalloc(&dec_DS_M, num_Elements * sizeof(int));
        err = cudaMalloc(&AS_M, num_Elements * sizeof(int));
        err = cudaMalloc(&dec_AS_M, num_Elements * sizeof(int));
        err = cudaMalloc((void**)&device_input_all, num_Elements * sizeof(double));
        err = cudaMalloc((void**)&device_decp_all, num_Elements * sizeof(double));
        
        cudaMemcpy(device_input_all, input_data, num_Elements * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(device_decp_all, global.data(), num_Elements * sizeof(double), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        std::ofstream fout("fixed_" + filename + std::to_string(bound) + "_" + compressor_id+ ".bin", std::ios::out | std::ios::binary);
        if (!fout) {
            std::cerr << "Error: cannot open file\n";
            return -1;
        }

        
        fout.write(reinterpret_cast<const char*>(global.data()), global.size() * sizeof(double));

        fout.close();

        if (err != cudaSuccess) {
            printf("Rank %d: before gathering: %s\n", rank, cudaGetErrorString(err));
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        std::cout<<num_Elements<<std::endl;
        iscriticle<double><<<gridSize, blockSize>>>(DS_M, AS_M, nullptr, nullptr, nullptr, device_input_all, device_input_all, edits, nullptr,
            width_host, height_host, depth_host, num_Elements, or_types, offset_x, offset_y, offset_z, width_host, height_host, rank, nullptr, q, bound, delta, ite, uTh, lTh, 1, 1, 1);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Rank %d: before compression failed: %s\n", rank, cudaGetErrorString(err));
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        iscriticle<double><<<gridSize, blockSize>>>(dec_DS_M, dec_AS_M, nullptr, nullptr, nullptr, device_decp_all, device_decp_all, edits, nullptr, 
            width_host, height_host, depth_host, num_Elements, or_types, offset_x, offset_y, offset_z, width_host, height_host, rank, nullptr, q, bound, delta, ite, uTh, lTh, 1, 1, 1);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Rank %d: before compression failed: %s\n", rank, cudaGetErrorString(err));
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        int *un_sign_as;
        cudaMalloc((void**)&un_sign_as, sizeof(int));
        cudaMemset(un_sign_as, 0, sizeof(int));

        
        PathCompression<<<gridSize, blockSize>>>(DS_M, num_Elements, width_host, height_host, depth_host, 0);
        cudaDeviceSynchronize();
        PathCompression<<<gridSize, blockSize>>>(AS_M, num_Elements, width_host, height_host, depth_host, 0);
        cudaDeviceSynchronize();

        PathCompression<<<gridSize, blockSize>>>(dec_DS_M, num_Elements, width_host, height_host, depth_host, 0);
        cudaDeviceSynchronize();
        PathCompression<<<gridSize, blockSize>>>(dec_AS_M, num_Elements, width_host, height_host, depth_host, 0);
        cudaDeviceSynchronize();
        
        unsigned int host_count_f_dir = 0, host_count_f_max = 0;
        int initialValue = 0;
        cudaMemcpyToSymbol(count_f_dir, &initialValue, sizeof(unsigned int));
        cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(unsigned int));
        err = cudaDeviceSynchronize();

        count_false_cases<<<gridSize, blockSize>>>(DS_M, dec_DS_M, AS_M, dec_AS_M, 
            num_Elements, width_host, height_host, depth_host);
        err = cudaDeviceSynchronize();

        cudaMemcpyFromSymbol(&host_count_f_dir, count_f_dir, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Rank %d: before compression failed: %s\n", rank, cudaGetErrorString(err));
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        std::cout<<"number of false cases: "<< host_count_f_dir<<", "<<host_count_f_max<<std::endl;

        
    }

    if(rank == 0){
        if(compression_mode == 0 && weakscaling == 0){
            cmpSize = std::filesystem::file_size(cpfilename);
            total_cmpsize = cmpSize;
        }
       
       
        std::uintmax_t original_dataSize = std::filesystem::file_size(file_path);
        double CR = double(original_dataSize) / total_cmpsize;
        double OCR = double(original_dataSize) / (total_cmpsize + total_storageOverhead);
        double edit_ratio = double(total_edited) / num_Elements;
        
        std::cout<<"total storage overhead is: "<<total_storageOverhead<<std::endl;
        
        std::cout<<"original CR is: "<< CR << std::endl;
        std::cout<<"overall CR is: "<< OCR << std::endl;
        std::cout<<"overall edit ratio is: "<< edit_ratio << " total_edited:" << total_edited<< std::endl;
        
        std::ofstream outFile3("./stat_result/Global_with_ite_result_"+filename+"_"+compressor_id+"_Maxonly_final"+std::to_string(size)+".txt", std::ios::app);
    
        if (!outFile3) {
            std::cerr << "Unable to open file for writing." << std::endl;
            return 0; 
        }

        outFile3 << std::to_string(bound)<<":" << std::endl;
        outFile3 << "iteration"<<":" <<global_max<< std::endl;
        outFile3 << std::setprecision(10)<< "absolute_error: "<< bound << std::endl;
        outFile3 << std::setprecision(10)<< "OCR: "<< OCR << std::endl;
        outFile3 << "threshold: "<< q << std::endl;
        outFile3 << "iteStep: "<< iteStep << std::endl;
        outFile3 <<std::setprecision(10)<< "CR: "<< CR << std::endl;
        

        
        outFile3 << std::setprecision(17)<<"edit_ratio: "<< edit_ratio << std::endl;
        outFile3 << std::setprecision(17)<<"compression_time: "<< compression_time<< std::endl;
        outFile3 << std::setprecision(17)<<"additional_time: "<< additional_time<< std::endl;
        outFile3 << std::setprecision(17)<<"Edits Time: "<< editsTime << std::endl;
        outFile3 << std::setprecision(17)<<"Storage_Overhead: "<< total_storageOverhead<< std::endl;
        outFile3 << std::setprecision(17)<<"Lower Threshold: "<< lTh << std::endl;
        outFile3 << std::setprecision(17)<<"Whole Time: "<< WholeTime << std::endl;
        outFile3 << std::setprecision(17)<<"Data Time: "<< dataTime << std::endl;
        for(auto ele: edits_time){
            outFile3 << std::setprecision(17)<<ele << ", ";
        }
        outFile3 << std::endl;
        outFile3 << "\n" << std::endl;

        
        outFile3.close();

        std::cout << "Variables have been appended to output.txt" << std::endl;

    }
    
    
    
    MPI_Finalize();
    return 0;
}


