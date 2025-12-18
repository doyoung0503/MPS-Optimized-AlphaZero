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

// ============================================================================
// MCTSNode: Memory-Safe Arena Node with Guaranteed 64-byte Alignment
// ============================================================================
// This struct MUST be exactly 384 bytes (6 cache lines) for safe arena indexing.
// Using explicit padding to ensure ARM64 alignment requirements are met.

struct alignas(64) MCTSNode {
    // === Block 1: Chess state (largest member first) ===
    Chess state;  // sizeof(Chess) = 280 bytes
    
    // === Block 2: Node metadata (grouped for cache efficiency) ===
    int player;                         // 4 bytes - offset 280
    int move_from_parent;               // 4 bytes - offset 284
    
    // === Block 3: Atomic counters (naturally aligned) ===
    std::atomic<int> visit_count{0};    // 4 bytes - offset 288
    std::atomic<int> virtual_loss{0};   // 4 bytes - offset 292
    std::atomic<float> value_sum{0.0f}; // 4 bytes - offset 296
    
    // === Block 4: Tree topology (64-bit indices for >2GB addressing) ===
    float prior = 0.0f;                 // 4 bytes - offset 300
    int64_t parent_index = -1;          // 8 bytes - offset 304 (changed from int32_t)
    int64_t first_child_index = -1;     // 8 bytes - offset 312 (changed from int32_t)
    
    // === Block 5: Expansion state ===
    std::atomic<int> expansion_state{0}; // 4 bytes - offset 320
    uint32_t num_children = 0;           // 4 bytes - offset 324 (changed from uint16_t for alignment)
    bool is_terminal = false;            // 1 byte  - offset 328
    
    // === Block 6: Terminal value (4-byte aligned) ===
    // Padding to align terminal_value to 4-byte boundary
    char _align_pad1[3] = {0};           // 3 bytes - offset 329 (pad to 332)
    float terminal_value = 0.0f;        // 4 bytes - offset 332
    
    // === Final Padding: Ensure total size is exactly 384 bytes ===
    // Current offset: 336 bytes. Need: 384 - 336 = 48 bytes padding
    char _padding[48];
    
    // Default constructor (must be trivial for POD-like behavior)
    MCTSNode() = default;
    
    // Init helper (called after placement new)
    // Uses int64_t for parent index to support >2GB arenas
    void init(const Chess& s, int p, int64_t par, int m, float pri) {
        state = s;
        player = p;
        parent_index = par;
        move_from_parent = m;
        prior = pri;
        visit_count.store(0, std::memory_order_relaxed);
        virtual_loss.store(0, std::memory_order_relaxed);
        value_sum.store(0.0f, std::memory_order_relaxed);
        first_child_index = -1;
        num_children = 0;
        expansion_state.store(0, std::memory_order_relaxed);
        is_terminal = false;
        terminal_value = 0.0f;
    }
};

// ============================================================================
// Compile-Time Verification: CRITICAL for Arena Safety
// ============================================================================
static_assert(sizeof(MCTSNode) == 384, "MCTSNode must be exactly 384 bytes");
static_assert(sizeof(MCTSNode) % 64 == 0, "MCTSNode must be 64-byte aligned for cache lines");
static_assert(alignof(MCTSNode) == 64, "MCTSNode alignment must be 64 bytes");

// 64-bit node indices for arenas > 2GB
struct InferenceRequest {
    size_t node_index;    // Changed from int to size_t
    int game_index; 
};

class MCTS {
public:
    MCTS(std::shared_ptr<AlphaZeroNet> net, int sims, float cpuct, int batch_size, torch::Device dev);
    ~MCTS();

    void reset_pool(); // Reset arena pointer
    size_t create_root(const Chess& state);  // Returns size_t index
    std::vector<std::vector<float>> search_async(const std::vector<size_t>& root_indices);  // Takes size_t indices
    int select_action(size_t root_index, float temp);  // Takes size_t index
    MCTSNode& get_node(size_t index);  // Takes size_t index

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
    
    size_t alloc_nodes(size_t count);  // Returns size_t index
    
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
    const std::vector<size_t>* current_roots = nullptr;  // Changed to size_t
    std::atomic<int>* current_sims_remaining = nullptr;
    std::atomic<int> active_workers{0};
    
    // Persistent GPU Buffer
    torch::Tensor gpu_input_buffer;
    
    // Loops
    void worker_loop(int worker_id);
    void inferencer_loop();
    
    // Logic
    void expand_node(size_t node_idx, const std::vector<float>& policy, float value);  // Changed to size_t
    float ucb(const MCTSNode& node, const MCTSNode& parent) const;
};
