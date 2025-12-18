#pragma once
#include "Chess.h"
#include "Network.h"
#include <vector>
#include <atomic>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <cmath>

// Index-based Node (Memory-Safe for Arena Allocator)
// alignas(64) + padding ensures each node starts on cache line boundary
struct alignas(64) MCTSNode {
    Chess state; // ~280 bytes (std::array based, POD-safe)
    int player; // 4
    int move_from_parent; // 4
    
    // Atomics for Lock-free / Fine-grained locking
    std::atomic<int> visit_count{0};    // 4 (but aligned to 4)
    std::atomic<int> virtual_loss{0};   // 4
    std::atomic<float> value_sum{0.0f}; // 4
    
    float prior = 0.0f; // 4
    
    // Tree Topology via Indices
    int32_t parent_index = -1;     // 4
    int32_t first_child_index = -1; // 4
    uint16_t num_children = 0;      // 2
    
    // 0: Unexpanded, 1: Expanding, 2: Expanded
    std::atomic<int> expansion_state{0}; // 4
    
    bool is_terminal = false;   // 1
    float terminal_value = 0.0f; // 4
    
    // Explicit padding to ensure sizeof(MCTSNode) is exactly 384 bytes (6 * 64)
    // Current rough size: 280 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 + 4 + 1 + 4 = ~323 bytes
    // Pad to 384 (6 cache lines) for clean alignment
    char _padding[384 - 280 - 4*7 - 2 - 1 - 4]; // Calculate remaining

    MCTSNode() = default;
    
    // Init helper (placement new uses this)
    void init(const Chess& s, int p, int32_t par, int m, float pri) {
        state = s;
        player = p;
        parent_index = par;
        move_from_parent = m;
        prior = pri;
        visit_count = 0;
        virtual_loss = 0;
        value_sum = 0.0f;
        first_child_index = -1;
        num_children = 0;
        expansion_state = 0;
        is_terminal = false;
        terminal_value = 0.0f;
    }
};

// Compile-time verification: MCTSNode must be 64-byte aligned and multiple of 64
static_assert(sizeof(MCTSNode) % 64 == 0, "MCTSNode size must be multiple of 64 bytes for arena alignment");
static_assert(alignof(MCTSNode) == 64, "MCTSNode must be 64-byte aligned");

struct InferenceRequest {
    int node_index;
    int game_index; 
};

class MCTS {
public:
    MCTS(std::shared_ptr<AlphaZeroNet> net, int sims, float cpuct, int batch_size, torch::Device dev);
    ~MCTS();

    void reset_pool(); // Reset arena pointer
    int create_root(const Chess& state);
    std::vector<std::vector<float>> search_async(const std::vector<int>& root_indices);
    int select_action(int root_index, float temp);
    MCTSNode& get_node(int index);

private:
    std::shared_ptr<AlphaZeroNet> model;
    int num_simulations;
    float c_puct;
    int max_batch_size;
    torch::Device device;
    
    // Arena Allocator
    MCTSNode* node_arena = nullptr;
    size_t arena_capacity = 0;
    std::atomic<size_t> allocated_count{0};
    
    int alloc_nodes(int count);
    
    // Thread Pool (Persistent)
    std::vector<std::thread> worker_threads;
    std::thread inference_thread_handle;
    
    std::atomic<bool> stop_signal{false};
    
    // Inference Coordination
    std::queue<InferenceRequest> inference_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    
    // Worker Coordination
    std::mutex work_mutex;
    std::condition_variable work_cv;
    std::condition_variable main_cv; // To wake main thread
    
    // Shared Job State
    bool search_in_progress = false;
    const std::vector<int>* current_roots = nullptr;
    std::atomic<int>* current_sims_remaining = nullptr;
    std::atomic<int> active_workers{0};
    
    // Persistent GPU Buffer
    torch::Tensor gpu_input_buffer;
    
    // Loops
    void worker_loop(int worker_id);
    void inferencer_loop();
    
    // Logic
    void expand_node(int node_idx, const std::vector<float>& policy, float value);
    float ucb(const MCTSNode& node, const MCTSNode& parent) const;
};
