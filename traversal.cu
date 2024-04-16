#include <climits>
#include <stdio.h>
#define N_VERTICES 9
#define N_EDGES 15
#define THREADS_PER_BLOCK 1

struct CSRGraph {
  unsigned int numVertices;
  unsigned int srcPtrs[N_VERTICES + 1];
  unsigned int dst[N_EDGES];
};

typedef struct CSRGraph graph_t;

graph_t *init_graph() {
  graph_t *g = (graph_t *)malloc(sizeof(graph_t));
  g->numVertices = N_VERTICES;

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
  printf("graph n_vertices = %u\n", g->numVertices);
  printf("src pointers: ");

  for (unsigned int i = 0; i < g->numVertices + 1; ++i) {
    printf("%u ", g->srcPtrs[i]);
  }

  printf("\n");
  printf("dst: ");

  for (unsigned int i = 0; i < N_EDGES; ++i) {
    printf("%u ", g->dst[i]);
  }

  printf("\n");
}

__global__ void bfs_kernel(graph_t *graph, unsigned int *level,
                           unsigned int *new_vertex_visited,
                           unsigned int *curr_level) {
  unsigned int vertex = blockDim.x * blockIdx.x + threadIdx.x;

  if (vertex < graph->numVertices) {

    if (level[vertex] == *curr_level - 1) {
      printf("Iterating over all edges associated with vertex id = %u\n",
             vertex);

      // printf("Src Ptrs size = %d\n", );
      printf("Vertex begins at edge id = %u\n", graph->srcPtrs[vertex]);
      printf("Vertex ends at edge id = %u\n", graph->srcPtrs[vertex + 1]);

      for (unsigned int edge = graph->srcPtrs[vertex];
           edge < graph->srcPtrs[vertex + 1]; ++edge) {
        printf("edge id = %u\n", edge);
        unsigned int neighbor = graph->dst[edge];
        printf("neighbor id = %u\n", neighbor);
        if (level[neighbor] == UINT_MAX) { // neighbor not visited

          printf("new neighbor id added to level %u = %u\n", *curr_level,
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
  unsigned int *level, *new_visited, *curr_level;

  graph_t *g_dev;
  unsigned int *level_dev, *new_visited_dev, *curr_level_dev;

  // Init BFS variables
  level = (unsigned int *)malloc(N_VERTICES * sizeof(unsigned int));
  curr_level = (unsigned int *)malloc(sizeof(unsigned int));
  new_visited = (unsigned int *)malloc(sizeof(unsigned int));

  level[0] = 0;
  for (int i = 1; i < N_VERTICES; ++i) {
    level[i] = INT_MAX;
  }
  *curr_level = 1;
  *new_visited = 0;

  cudaMalloc((void **)&g_dev, sizeof(graph_t));
  cudaMalloc((void **)&level_dev, N_VERTICES * sizeof(unsigned int));
  cudaMalloc((void **)&curr_level_dev, sizeof(unsigned int));
  cudaMalloc((void **)&new_visited_dev, sizeof(unsigned int));

  cudaMemcpy(g_dev, g, sizeof(graph_t), cudaMemcpyHostToDevice);
  cudaMemcpy(level_dev, level, N_VERTICES * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(curr_level_dev, curr_level, sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(new_visited_dev, new_visited, sizeof(unsigned int),
             cudaMemcpyHostToDevice);

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
