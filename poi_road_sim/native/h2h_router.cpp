// Directed shortest paths with the supplied undirected H2H labels as an A* lower bound.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
constexpr int INF = 999999999;
thread_local std::string last_error;

int read_int(std::ifstream& in) {
    int value;
    if (!in.read(reinterpret_cast<char*>(&value), sizeof(value)))
        throw std::runtime_error("truncated H2H index");
    return value;
}

std::vector<int> read_vector(std::ifstream& in, int size) {
    if (size < 0 || size > 100000000) throw std::runtime_error("invalid H2H array size");
    std::vector<int> values(size);
    if (size && !in.read(reinterpret_cast<char*>(values.data()), size * sizeof(int)))
        throw std::runtime_error("truncated H2H index");
    return values;
}

struct Index {
    int nodes, root;
    std::vector<int> height, belong, to_rmq;
    std::vector<std::vector<int>> rmq, pos, dis, pos2;

    explicit Index(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open H2H index");
        nodes = read_int(in);
        int trees = read_int(in);
        if (nodes < 1 || trees < 1 || trees > nodes)
            throw std::runtime_error("invalid H2H header");
        height = read_vector(in, trees);
        belong = read_vector(in, nodes + 1);
        to_rmq = read_vector(in, nodes + 1);
        int levels = read_int(in);
        int euler_size = read_int(in);
        if (levels < 1 || levels > 32 || euler_size < 1 || euler_size > nodes * 2 + 1)
            throw std::runtime_error("invalid H2H RMQ header");
        for (int i = 0; i < levels; ++i) rmq.push_back(read_vector(in, read_int(in)));
        root = read_int(in);
        if (root < 0 || root >= trees) throw std::runtime_error("invalid H2H root");
        for (int i = 0; i < trees; ++i) read_vector(in, read_int(in)); // child lists
        pos.resize(trees);
        dis.resize(trees);
        pos2.resize(trees);
        for (int i = 0; i < trees; ++i) {
            int node = read_int(in);
            if (node < 0 || node >= trees) throw std::runtime_error("invalid H2H tree node");
            pos[node] = read_vector(in, read_int(in));
            dis[node] = read_vector(in, read_int(in));
        }
        for (int i = 0; i < trees; ++i) {
            int node = read_int(in);
            if (node < 0 || node >= trees) throw std::runtime_error("invalid H2H tree node");
            pos2[node] = read_vector(in, read_int(in));
        }
    }

    int distance(int p, int q) const {
        if (p < 1 || q < 1 || p > nodes || q > nodes) return INF;
        if (p == q) return 0;
        int x = belong[p], y = belong[q];
        if (x < 0 || y < 0 || x >= int(height.size()) || y >= int(height.size()))
            throw std::runtime_error("invalid H2H vertex label");
        int left = to_rmq[x], right = to_rmq[y];
        if (left > right) std::swap(left, right);
        int span = right - left + 1, level = 0;
        while ((1 << (level + 1)) <= span) ++level;
        int block = 1 << level;
        if (level >= int(rmq.size()) || left >= int(rmq[level].size()) ||
            right - block + 1 >= int(rmq[level].size()))
            throw std::runtime_error("invalid H2H RMQ label");
        int a = rmq[level][left], b = rmq[level][right - block + 1];
        int lca = height[a] < height[b] ? a : b;
        if (lca == x || lca == y) {
            if (lca == y) std::swap(x, y);
            if (pos[x].empty() || pos[x].back() < 0 || pos[x].back() >= int(dis[y].size()))
                throw std::runtime_error("invalid H2H ancestor label");
            return dis[y][pos[x].back()];
        }
        int result = INF;
        for (int i : pos2[lca]) {
            if (i < 0 || i >= int(dis[x].size()) || i >= int(dis[y].size()))
                throw std::runtime_error("invalid H2H distance label");
            result = std::min(result, dis[x][i] + dis[y][i]);
        }
        return result;
    }
};

struct Edge { int from, to, id; double cost, free_cost, default_cost; };
struct Router {
    Index index;
    int node_count;
    std::vector<int> local, offsets;
    std::vector<Edge> edges;
    std::vector<int> edge_location;
    int bound_violations = 0;

    Router(const std::string& index_file, const std::string& map_file,
           const std::string& roads_file) : index(index_file) {
        std::ifstream map(map_file);
        int mapped;
        if (!(map >> node_count >> mapped) || node_count < 1 || mapped != index.nodes)
            throw std::runtime_error("node mapping does not match H2H index");
        local.assign(node_count + 1, 0);
        int original, dense;
        for (int i = 0; i < mapped; ++i) {
            if (!(map >> original >> dense) || original < 1 || original > node_count ||
                dense < 1 || dense > mapped || local[original])
                throw std::runtime_error("invalid H2H node mapping");
            local[original] = dense;
        }
        std::ifstream roads(roads_file);
        int count;
        if (!(roads >> count) || count < 0) throw std::runtime_error("invalid directed edge header");
        edges.reserve(count);
        for (int i = 0; i < count; ++i) {
            Edge edge;
            if (!(roads >> edge.id >> edge.from >> edge.to >> edge.cost >> edge.free_cost) ||
                edge.from < 1 || edge.from > node_count || edge.to < 1 ||
                edge.to > node_count || !std::isfinite(edge.cost) || edge.cost <= 0 ||
                !std::isfinite(edge.free_cost) || edge.free_cost <= 0 ||
                edge.free_cost > edge.cost)
                throw std::runtime_error("invalid directed edge data");
            edge.default_cost = edge.cost;
            edges.push_back(edge);
        }
        std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
            return a.from < b.from;
        });
        int max_id = 0;
        for (const auto& edge : edges) {
            if (edge.id < 0) throw std::runtime_error("negative directed edge ID");
            max_id = std::max(max_id, edge.id);
        }
        if (max_id > count * 16 + 1000000)
            throw std::runtime_error("edge IDs are too sparse for the route index");
        edge_location.assign(max_id + 1, -1);
        for (size_t i = 0; i < edges.size(); ++i) {
            if (edge_location[edges[i].id] != -1)
                throw std::runtime_error("duplicate directed edge ID");
            edge_location[edges[i].id] = int(i);
        }
        offsets.assign(node_count + 2, 0);
        for (const auto& edge : edges) ++offsets[edge.from + 1];
        for (size_t i = 1; i < offsets.size(); ++i) offsets[i] += offsets[i - 1];
    }

    double lower_bound(int from, int to) const {
        if (from < 1 || to < 1 || from > node_count || to > node_count)
            throw std::runtime_error("node ID is outside road graph");
        if (bound_violations || !local[from] || !local[to]) return 0.0;
        int units = index.distance(local[from], local[to]);
        return units >= INF ? 0.0 : units / 100.0;
    }

    void update_weights(const int* ids, const double* weights, int count) {
        if (count < 0) throw std::runtime_error("negative weight update count");
        for (int i = 0; i < count; ++i) {
            int id = ids[i];
            if (id < 0 || id >= int(edge_location.size()) || edge_location[id] == -1)
                throw std::runtime_error("unknown edge ID in weight update");
            const Edge& edge = edges[edge_location[id]];
            if (!std::isfinite(weights[i]) || weights[i] <= 0)
                throw std::runtime_error("road time must be finite and positive");
        }
        for (int i = 0; i < count; ++i) {
            Edge& edge = edges[edge_location[ids[i]]];
            bound_violations -= edge.cost + 1e-9 < edge.free_cost;
            edge.cost = weights[i];
            bound_violations += edge.cost + 1e-9 < edge.free_cost;
        }
    }

    void reset_weights() {
        for (auto& edge : edges) edge.cost = edge.default_cost;
        bound_violations = 0;
    }

    bool path(int from, int to, std::vector<int>& edge_ids,
              std::vector<int>& nodes, double& cost, bool reconstruct = true,
              bool use_index = true) const {
        if (from < 1 || to < 1 || from > node_count || to > node_count)
            throw std::runtime_error("node ID is outside road graph");
        if (from == to) {
            cost = 0;
            if (reconstruct) nodes = {from};
            return true;
        }
        struct Item { double estimate, cost; int node; };
        auto order = [](const Item& a, const Item& b) { return a.estimate > b.estimate; };
        std::priority_queue<Item, std::vector<Item>, decltype(order)> queue(order);
        std::unordered_map<int, double> best;
        std::unordered_map<int, int> parent;
        best[from] = 0;
        queue.push({use_index ? lower_bound(from, to) : 0.0, 0, from});
        while (!queue.empty()) {
            Item item = queue.top(); queue.pop();
            if (item.cost > best.at(item.node) + 1e-9) continue;
            if (item.node == to) {
                cost = item.cost;
                if (!reconstruct) return true;
                int current = to;
                nodes.push_back(current);
                while (current != from) {
                    int edge_index = parent.at(current);
                    const Edge& edge = edges[edge_index];
                    edge_ids.push_back(edge.id);
                    current = edge.from;
                    nodes.push_back(current);
                }
                std::reverse(nodes.begin(), nodes.end());
                std::reverse(edge_ids.begin(), edge_ids.end());
                return true;
            }
            for (int i = offsets[item.node]; i < offsets[item.node + 1]; ++i) {
                const Edge& edge = edges[i];
                double next = item.cost + edge.cost;
                auto found = best.find(edge.to);
                if (found == best.end() || next < found->second - 1e-9) {
                    best[edge.to] = next;
                    if (reconstruct) parent[edge.to] = i;
                    queue.push({next + (use_index ? lower_bound(edge.to, to) : 0.0), next, edge.to});
                }
            }
        }
        return false;
    }
};
} // namespace

extern "C" {
const char* h2h_last_error() { return last_error.c_str(); }
void* h2h_open(const char* index, const char* mapping, const char* roads) {
    try { last_error.clear(); return new Router(index, mapping, roads); }
    catch (const std::exception& error) { last_error = error.what(); return nullptr; }
}
void h2h_close(void* handle) { delete static_cast<Router*>(handle); }
double h2h_lower_bound(void* handle, int from, int to) {
    try { last_error.clear(); return static_cast<Router*>(handle)->lower_bound(from, to); }
    catch (const std::exception& error) { last_error = error.what(); return -1; }
}
int h2h_update_weights(void* handle, const int* ids, const double* weights, int count) {
    try {
        last_error.clear();
        static_cast<Router*>(handle)->update_weights(ids, weights, count);
        return 0;
    } catch (const std::exception& error) { last_error = error.what(); return -1; }
}
int h2h_reset_weights(void* handle) {
    try { last_error.clear(); static_cast<Router*>(handle)->reset_weights(); return 0; }
    catch (const std::exception& error) { last_error = error.what(); return -1; }
}
int h2h_path(void* handle, int from, int to, int** edge_ids, int** nodes,
             int* edge_count, double* cost) {
    try {
        last_error.clear();
        std::vector<int> edges, vertices;
        if (!static_cast<Router*>(handle)->path(from, to, edges, vertices, *cost)) return 0;
        *edge_count = int(edges.size());
        *edge_ids = new int[edges.size()];
        *nodes = new int[vertices.size()];
        std::copy(edges.begin(), edges.end(), *edge_ids);
        std::copy(vertices.begin(), vertices.end(), *nodes);
        return 1;
    } catch (const std::exception& error) { last_error = error.what(); return -1; }
}
int h2h_distance(void* handle, int from, int to, double* cost) {
    try {
        last_error.clear();
        std::vector<int> edges, nodes;
        return static_cast<Router*>(handle)->path(from, to, edges, nodes, *cost, false) ? 1 : 0;
    } catch (const std::exception& error) { last_error = error.what(); return -1; }
}
int h2h_plain_distance(void* handle, int from, int to, double* cost) {
    try {
        last_error.clear();
        std::vector<int> edges, nodes;
        return static_cast<Router*>(handle)->path(from, to, edges, nodes, *cost, false, false) ? 1 : 0;
    } catch (const std::exception& error) { last_error = error.what(); return -1; }
}
void h2h_free(int* data) { delete[] data; }
}
