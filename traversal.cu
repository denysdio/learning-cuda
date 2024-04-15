#include <climits>
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

__global__ void bfs_kernel(graph_t *graph, int *level, int *new_vertex_visited,
                           int *curr_level) {
  int vertex = blockDim.x * blockIdx.x + threadIdx.x;

  if (vertex < graph->numVertices) {

    if (level[vertex] == *curr_level - 1) {
      printf("Iterating over all edges associated with vertex id = %d\n",
             vertex);
      graph->srcPtrs = (int *)malloc((N_VERTICES + 1) * sizeof(int));
      graph->dst = (int *)malloc(N_EDGES * sizeof(int));

      graph->srcPtrs[0] = 0;
      graph->srcPtrs[1] = 2;
      graph->srcPtrs[2] = 4;
      graph->srcPtrs[3] = 7;
      graph->srcPtrs[4] = 9;
      graph->srcPtrs[5] = 11;
      graph->srcPtrs[6] = 12;
      graph->srcPtrs[7] = 13;
      graph->srcPtrs[8] = 15;
      graph->srcPtrs[9] = 15;

      graph->dst[0] = 1;
      graph->dst[1] = 2;
      graph->dst[2] = 3;
      graph->dst[3] = 4;
      graph->dst[4] = 5;
      graph->dst[5] = 6;
      graph->dst[6] = 7;
      graph->dst[7] = 4;
      graph->dst[8] = 8;
      graph->dst[9] = 5;
      graph->dst[10] = 8;
      graph->dst[11] = 6;
      graph->dst[12] = 8;
      graph->dst[13] = 0;
      graph->dst[14] = 6;

      // printf("Src Ptrs size = %d\n", );
      printf("Vertex begins at edge id = %d\n", graph->srcPtrs[vertex]);
      printf("Vertex ends at edge id = %d\n", graph->srcPtrs[vertex + 1]);

      for (int edge = graph->srcPtrs[vertex]; edge < graph->srcPtrs[vertex + 1];
           ++edge) {
        printf("edge id = %d\n", edge);
        int neighbor = graph->dst[edge];
        printf("neighbor id = %d\n", neighbor);
        if (level[neighbor] == INT_MAX) { // neighbor not visited

          printf("new neighbor id added to level %d = %d\n", *curr_level,
                 neighbor);
          level[neighbor] = *curr_level;
          *new_vertex_visited = 1;
        }
      }
    }
  }
}

int main(void) {
  graph_t *g = init_graph();
  int *level, *new_visited, *curr_level;

  graph_t *g_dev;
  int *level_dev, *new_visited_dev, *curr_level_dev;

  // Init BFS variables
  level = (int *)malloc(N_VERTICES * sizeof(int));
  curr_level = (int *)malloc(sizeof(int));
  new_visited = (int *)malloc(sizeof(int));

  level[0] = 0;
  for (int i = 1; i < N_VERTICES; ++i) {
    level[i] = INT_MAX;
  }
  *curr_level = 1;
  *new_visited = 0;

  cudaMalloc((void **)&g_dev, sizeof(graph_t));
  cudaMalloc((void **)&level_dev, N_VERTICES * sizeof(int));
  cudaMalloc((void **)&curr_level_dev, sizeof(int));
  cudaMalloc((void **)&new_visited_dev, sizeof(int));

  cudaMemcpy(g_dev, g, sizeof(graph_t), cudaMemcpyHostToDevice);
  cudaMemcpy(level_dev, level, N_VERTICES * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(curr_level_dev, curr_level, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(new_visited_dev, new_visited, sizeof(int), cudaMemcpyHostToDevice);

  bfs_kernel<<<N_VERTICES / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
      g_dev, level_dev, new_visited_dev, curr_level_dev);

  free(g);
  free(level);
  free(new_visited);
  free(curr_level);

  cudaFree(g_dev);
  cudaFree(level_dev);
  cudaFree(new_visited_dev);
  cudaFree(curr_level_dev);

  // print_graph(g);
  return 0;
}
