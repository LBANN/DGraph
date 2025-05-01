#include <iostream>
#include <vector>
#include <tuple>
#include <random>
#include <chrono>
#include <set>
#include <algorithm>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <sstream>
// using adj_list = std::unordered_map<unsigned long, std::set<unsigned long>>;
// Function to generate a random connected graph with a maximum degree constraint
// and return it as an adjacency list in an unordered_map (single-threaded).

bool find_in_adj_list(const std::vector<unsigned long> &adj_list, unsigned long vertex)
{
  return std::find(adj_list.begin(), adj_list.end(), vertex) != adj_list.end();
}

using adj_list = std::vector<std::vector<unsigned long>>;

std::tuple<adj_list, unsigned long> generate_graph(
    const unsigned long &num_vertices,
    const int &max_degree)
{
  adj_list adjacency_map;

  if (num_vertices == 0)
  {
    std::cerr << "Warning: Number of vertices is 0. Returning empty map." << std::endl;
    return std::tuple<adj_list, unsigned long>(adjacency_map, 0);
  }

  if (max_degree <= 0)
  {
    std::cerr << "Error: Maximum degree must be greater than 0." << std::endl;
    // Depending on requirements, might return empty or throw exception
    return std::tuple<adj_list, unsigned long>(adjacency_map, 0);
  }

  // Keep track of the current degree of each vertex.
  std::vector<int> degree(num_vertices, 0);

  adjacency_map.reserve(num_vertices);  
  unsigned seed = std::random_device()();
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<unsigned long> dist_vertices(0, num_vertices - 1);

  // 1. Build a random spanning tree to ensure connectivity (for num_vertices > 1).

  std::vector<unsigned long> visited;
  visited.reserve(num_vertices);
  visited.push_back(0); // Start with vertex 0
  std::vector<unsigned long> unvisited;
  unvisited.reserve(num_vertices - 1);

  for (unsigned long i = 0; i < num_vertices; ++i)
  {
    adjacency_map.push_back(std::vector<unsigned long>(max_degree, num_vertices));
  }
  for (unsigned long i = 1; i < num_vertices; ++i)
  {
    unvisited.push_back(i);
  }

  std::shuffle(unvisited.begin(), unvisited.end(), rng);

  long num_edges = 0;
  auto num_vertex_processed = 1;
  std::cout << "Building spanning tree..." << std::endl;

  for (auto u : unvisited)
  {
    // Randomly select a vertex from the visited set
    std::uniform_int_distribution<unsigned long>
        dist_visited(0, visited.size() - 1);
    unsigned long v_idx = dist_visited(rng);
    unsigned long v = visited[v_idx];
    auto max_degree_check = std::max(degree[u], degree[v]) < max_degree;

    while (!max_degree_check)
    {
      // If the edge already exists or max degree is exceeded, try again
      v_idx = dist_visited(rng);
      v = visited[v_idx];
      max_degree_check = std::max(degree[u], degree[v]) < max_degree;
    }

    adjacency_map[u][degree[u]] = v;
    adjacency_map[v][degree[v]] = u;

    // std::cout << std::endl;
    degree[u]++;
    degree[v]++;

    num_edges++;
    num_vertex_processed++;

    visited.push_back(u);
  }

  


  std::cout << "Number of edges in the spanning tree: " << num_edges << std::endl;
  if (visited.size() != num_vertices)
  {
    std::cerr << "Warning: Could not connect all vertices while building spanning tree (possibly due to low max_degree). Graph might be disconnected." << std::endl;
  }

  visited.clear();
  unvisited.clear();

  // Graph is connected with all edges having at least one vertex
  // and a maximum degree of max_degree.

  // Add additional random edges between vertices (will not exceed max_degree).
  // Graph will of course remain connected.

  // heuristic
  const auto max_new_edges = num_edges * 2;
  const auto min_new_edges = num_edges / 2;

  // chose a number between min_new_edges and max_new_edges
  std::uniform_int_distribution<unsigned long> dist_edges(min_new_edges, max_new_edges);
  const auto num_edges_to_add = dist_edges(rng);

  unsigned long max_possible_edges = num_vertices * (max_degree) / 4;

  unsigned long current_edges_count = num_edges;
  // Attempt to add more edges for a reasonable number of times

  const auto max_possible_edges = num_vertices * (num_vertices - 1) / 2;

  std::cout << "Total edge attempts: " << num_edges_to_add << std::endl;
  unsigned int attempts = 0;
  while (attempts < num_edges_to_add)
  {
    attempts++; // Count attempt even if unsuccessful
    unsigned long u = dist_vertices(rng);
    unsigned long v = dist_vertices(rng);

    if (u == v)
    {
      continue; // No self-loops
    }

    unsigned long first = std::min(u, v);
    unsigned long second = std::max(u, v);

    const auto &adj_list = adjacency_map[first];
    // Check if the edge already exists

    // const auto not_in_adj_list = adj_list.find(second) == adj_list.end();
    const auto in_adj_list = find_in_adj_list(adj_list, second);
    // Check if edge exists and if adding it violates max_degree
    if (!in_adj_list && std::max(degree[u], degree[v]) < max_degree)
    {
      adjacency_map[u][degree[u]] = v;
      adjacency_map[v][degree[v]] = u; // Add the edge in both directions
      degree[u]++;
      degree[v]++;
      current_edges_count++;
    }
    if (current_edges_count >= max_possible_edges)
    { // Stop if we reach the maximum possible edges
      break;
    }
  }

  return std::tuple<adj_list, unsigned long>(adjacency_map, current_edges_count);
}

void write_graph_file(
    const adj_list &graph,
    const std::string &filename,
    const unsigned long &num_vertices,
    const unsigned long &num_edges)
{
  std::cout << "Write_graph_file called for file " << filename << std::endl;
  // Example: Simulate writing to a file
  std::ofstream outfile(filename);

  if (!outfile.is_open())
  {
    std::cerr << "Error opening file " << filename << " for writing." << std::endl;
    return;
  }

  outfile << num_vertices << " " << num_edges << std::endl;
  for (unsigned long i = 0; i < num_vertices; ++i)
  {
    const auto &neighbors = graph.at(i);
    outfile << neighbors[0] + 1;

    for (auto it = neighbors.begin() + 1; it != neighbors.end(); ++it)
    {
      if (*it < num_vertices)
      {
        outfile << " " << *it + 1;
      }
    }
    outfile << '\n';
  }
  outfile.close();
}

int main(int argc, char *argv[])
{
  // Check if the correct number of command-line arguments is provided
  if (argc != 4)
  {
    std::cerr << "Usage: " << argv[0] << " <num_vertices> <max_degree> <output_filename>" << std::endl;
    return 1; // Indicate an error
  }

  unsigned long num_vertices;
  unsigned int max_degree_uint;
  std::string output_filename;

  // Parse the number of vertices
  try
  {
    // Use std::stoul for unsigned long
    num_vertices = std::stoul(argv[1]);
    // Optional: Add a check for extremely large values if needed
    if (num_vertices == 0)
    {
      std::cerr << "Error: Number of vertices must be greater than 0." << std::endl;
      return 1;
    }
  }
  catch (const std::invalid_argument &ia)
  {
    std::cerr << "Error: Invalid number of vertices provided: " << ia.what() << std::endl;
    return 1;
  }
  catch (const std::out_of_range &oor)
  {
    std::cerr << "Error: Number of vertices out of range: " << oor.what() << std::endl;
    return 1;
  }

  // Parse the maximum degree
  try
  {
    // Use std::stoi for int, then cast to unsigned int
    int max_degree_int = std::stoi(argv[2]);
    if (max_degree_int < 1)
    {
      std::cerr << "Error: Maximum degree must be at least 1." << std::endl;
      return 1;
    }
    // Check if the integer value fits into an unsigned int
    if (max_degree_int > 10)
    {
      std::cerr << "Error: Maximum degree value too large for unsigned int." << std::endl;
      return 1;
    }
    max_degree_uint = static_cast<unsigned int>(max_degree_int);
  }
  catch (const std::invalid_argument &ia)
  {
    std::cerr << "Error: Invalid maximum degree provided: " << ia.what() << std::endl;
    return 1;
  }
  catch (const std::out_of_range &oor)
  {
    std::cerr << "Error: Maximum degree out of range: " << oor.what() << std::endl;
    return 1;
  }

  // The third argument is the filename
  output_filename = argv[3];

  // Cast max_degree_uint to int for the generate_graph_map function signature
  int max_degree_int_for_func = static_cast<int>(max_degree_uint);

  const auto graph_adj_list = generate_graph(num_vertices, max_degree_int_for_func);

  // Print the generated adjacency list
  std::cout << "Generated Graph Adjacency List:" << std::endl;
  // Iterate through vertices 0 to num_vertices-1 for consistent output order

  const auto &adj_list = std::get<0>(graph_adj_list);
  const auto &num_edges = std::get<1>(graph_adj_list);
  write_graph_file(adj_list, output_filename, num_vertices, num_edges);
  return 0;
}