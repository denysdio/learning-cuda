#include <stdio.h>
#define N_VERTICES 9
#define N_EDGES 15
#define THREADS_PER_BLOCK 1

struct CSRGraph {
  int numVertices;
  int *srcPtrs;
  int *dst;
};

typedef struct CSRGraph graph_t;

graph_t *init_graph() {
  graph_t *g = (graph_t *)malloc(sizeof(graph_t));
  g->numVertices = N_VERTICES;
  g->srcPtrs = (int *)malloc((N_VERTICES + 1) * sizeof(int));
  g->dst = (int *)malloc(N_EDGES * sizeof(int));

  g->srcPtrs[0] = 0;
  g->srcPtrs[1] = 2;
  g->srcPtrs[2] = 4;
  g->srcPtrs[3] = 7;
  g->srcPtrs[4] = 9;
  g->srcPtrs[5] = 11;
  g->srcPtrs[6] = 12;
  g->srcPtrs[7] = 13;
  g->srcPtrs[8] = 15;
  g->srcPtrs[9] = 15;

  g->dst[0] = 1;
  g->dst[1] = 2;
  g->dst[2] = 3;
  g->dst[3] = 4;
  g->dst[4] = 5;
  g->dst[5] = 6;
  g->dst[6] = 7;
  g->dst[7] = 4;
  g->dst[8] = 8;
  g->dst[9] = 5;
  g->dst[10] = 8;
  g->dst[11] = 6;
  g->dst[12] = 8;
  g->dst[13] = 0;
  g->dst[14] = 6;

  return g;
}

void print_graph(graph_t *g) {
  printf("graph n_vertices = %d\n", g->numVertices);
  printf("src pointers: ");

  for (int i = 0; i < g->numVertices + 1; ++i) {
    printf("%d ", g->srcPtrs[i]);
  }

  printf("\n");
  printf("dst: ");

  for (int i = 0; i < N_EDGES; ++i) {
    printf("%d ", g->dst[i]);
  }

  printf("\n");
}

__global__ void foo(graph_t *g, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < n) {
    printf("graph n_vertices = %d\n", g->numVertices);
  }
}

int main(void) {
  graph_t *g = init_graph();
  graph_t *g_dev;

  cudaMalloc((void **)&g_dev, sizeof(graph_t));
  cudaMemcpy(g_dev, g, sizeof(graph_t), cudaMemcpyHostToDevice);

  foo<<<N_VERTICES / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(g_dev, 1);

  free(g);
  cudaFree(g_dev);

  // print_graph(g);
  return 0;
}
