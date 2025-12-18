#include "MCTS.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <sys/mman.h> // For mmap
#include <unistd.h>
#include <random>

// --- Static Mirror Table ---
static std::vector<int> MIRROR_TABLE;
static std::once_flag mirror_flag;

void init_mirror_table() {
    MIRROR_TABLE.resize(4352);
    for(int i=0; i<4352; ++i) {
        Move m;
        m.from = i / 64; m.to = i % 64; m.promotion = 0;
        int f_r = m.from / 8; int f_c = m.from % 8;
        int t_r = m.to / 8; int t_c = m.to % 8;
        int from_new = (7 - f_r) * 8 + f_c;
        int to_new = (7 - t_r) * 8 + t_c;
        int mirrored = from_new * 64 + to_new;
        MIRROR_TABLE[i] = (mirrored < 4352) ? mirrored : i;
    }
    std::cout << "[MCTS] Mirror Table Initialized." << std::endl;
}

// --- MCTS Implementation ---

MCTS::MCTS(std::shared_ptr<AlphaZeroNet> net, int sims, float cpuct, int batch_size, torch::Device dev)
    : model(net), num_simulations(sims), c_puct(cpuct), max_batch_size(batch_size), device(dev) {
    
    std::call_once(mirror_flag, init_mirror_table);
    
    // Arena Allocator: Fixed MCTSNode array
    // M4 Pro 48GB Optimization: 80M nodes * 384 bytes = 30GB arena
    // sizeof(MCTSNode) = 384 bytes (verified at compile time)
    // All indices are now 64-bit (size_t) to prevent 2GB/4GB overflow
    size_t num_nodes = 80000000; // 80 million nodes (~30GB) for M4 Pro 48GB
    
    size_t size_bytes = static_cast<size_t>(num_nodes) * static_cast<size_t>(sizeof(MCTSNode));
    int prot = PROT_READ | PROT_WRITE;
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    
    std::cout << "[MCTS] sizeof(MCTSNode) = " << sizeof(MCTSNode) << " bytes" << std::endl;
    std::cout << "[MCTS] sizeof(Chess) = " << sizeof(Chess) << " bytes" << std::endl;
    std::cout << "[MCTS] Allocating " << (size_bytes / 1024 / 1024) << " MB Arena (" << num_nodes << " nodes)" << std::endl;
    
    node_arena = (MCTSNode*)mmap(NULL, size_bytes, prot, flags, -1, 0);
    if(node_arena == MAP_FAILED) {
        perror("mmap");
        std::cerr << "Failed to allocate Arena of size " << size_bytes << std::endl;
        exit(1);
    }
    
    arena_capacity = num_nodes;
    allocated_count = 0;
    
    std::cout << "[MCTS] Arena Allocated: " << (size_bytes / 1024 / 1024) << " MB (" << num_nodes << " nodes)" << std::endl;
    
    // Runtime Alignment Verification (Critical for ARM64)
    uintptr_t addr = reinterpret_cast<uintptr_t>(node_arena);
    if(addr % 64 != 0) {
        std::cerr << "[CRITICAL] Arena not aligned to 64 bytes! Addr: " << addr << std::endl;
        exit(1);
    }
    std::cout << "[MCTS] Arena Aligned (64 bytes). Addr: 0x" << std::hex << addr << std::dec << std::endl;
    
    // Verify node stride alignment
    if(sizeof(MCTSNode) % 64 != 0) {
        std::cerr << "[CRITICAL] MCTSNode size " << sizeof(MCTSNode) << " not 64-byte multiple!" << std::endl;
        exit(1);
    }
    
    // 2. Persistent GPU Buffer (increase for larger batches)
    gpu_input_buffer = torch::zeros({512, 14, 8, 8}, torch::kFloat).to(device);
    
    // =========================================================================
    // PRODUCTION CONFIGURATION (Phase 20)
    // =========================================================================
    // M4 Pro 48GB optimal settings after extensive Phase 16-19 testing:
    // - 4 workers: Proven stable for 23+ min, 12.5% arena (7.5M nodes)
    // - 8 workers: Faster (608 vs 284 moves/min) but less stable
    // - Recommendation: Use 4 workers for long training, 6-8 for benchmarks
    int num_workers = 4; // PRODUCTION: Conservative for unattended training
    
    std::cout << "[MCTS] Spawning " << num_workers << " persistent workers." << std::endl;
    
    stop_signal = false;
    
    inference_thread_handle = std::thread(&MCTS::inferencer_loop, this);
    for(int i=0; i<num_workers; ++i) {
        worker_threads.emplace_back(&MCTS::worker_loop, this, i);
    }
}

MCTS::~MCTS() {
    stop_signal = true;
    queue_cv.notify_all();
    {
        std::lock_guard<std::mutex> lock(work_mutex);
        work_cv.notify_all();
    }
    
    if(inference_thread_handle.joinable()) inference_thread_handle.join();
    for(auto& t : worker_threads) if(t.joinable()) t.join();
    
    // Free Arena
    if(node_arena) {
        munmap(node_arena, arena_capacity * sizeof(MCTSNode));
    }
}

void MCTS::reset_pool() {
    // Reset allocator cursor.
    // Old nodes are conceptually "freed" but we don't destruct them.
    // They will be overwritten by init().
    allocated_count = 0;
}

size_t MCTS::create_root(const Chess& state) {
    size_t idx = alloc_nodes(1);
    // Placement new to construct atomics properly
    new (&node_arena[idx]) MCTSNode(); 
    node_arena[idx].init(state, state.turn, -1, -1, 0.0f);
    
    // Add VL for root? No, worker loop adds it.
    
    return idx;
}

size_t MCTS::alloc_nodes(size_t count) {
    size_t idx = allocated_count.fetch_add(count);
    if(idx + count > arena_capacity) {
        std::cerr << "[CRITICAL] Arena Exhausted! Used " << idx << " of " << arena_capacity << std::endl;
        std::cerr << "[CRITICAL] Attempting to allocate " << count << " more nodes." << std::endl;
        exit(1);
    }
    
    // Periodic monitoring (every 500k nodes)
    size_t prev_milestone = (idx / 500000) * 500000;
    size_t curr_milestone = ((idx + count) / 500000) * 500000;
    if(curr_milestone > prev_milestone) {
        float usage_pct = 100.0f * (idx + count) / arena_capacity;
        std::cout << "[MCTS] Arena Usage: " << (idx + count) << " / " << arena_capacity 
                  << " (" << std::fixed << std::setprecision(1) << usage_pct << "%)" << std::endl;
    }
    
    return idx;  // Returns size_t, no truncation!
}

MCTSNode& MCTS::get_node(size_t index) {
    return node_arena[index];
}

// --- WORKER LOOP ---
void MCTS::worker_loop(int worker_id) {
    // Stride Calculation
    // We assume worker_threads is fully populated by the time this runs (started in ctor).
    // Accessing size() is safe since we don't modify the vector after start.
    int stride = worker_threads.size(); 
    if(stride == 0) stride = 1; // Safety
    
    while(!stop_signal) {
        {
            std::unique_lock<std::mutex> lock(work_mutex);
            work_cv.wait(lock, [&]{ return search_in_progress || stop_signal; });
        }
        
        if(stop_signal) return;
        
        // Job Received (using 64-bit indices)
        const std::vector<size_t>& roots = *current_roots;
        size_t num_games = roots.size();
        
        try {
            for(size_t game_idx = worker_id; game_idx < num_games; game_idx += stride) {
                size_t root_idx = roots[game_idx];
                if(root_idx >= arena_capacity) continue;  // size_t is always >= 0
                
                std::atomic<int>& remaining = current_sims_remaining[game_idx];
                
                while(remaining > 0) {
                    remaining--;
                    
                    size_t curr = root_idx;  // 64-bit index
                    // VL: Add to Root immediately (Descent phase)
                    node_arena[curr].virtual_loss++;
                    
                    std::vector<size_t> path;  // 64-bit indices
                    path.reserve(64);
                    path.push_back(curr);
                    
                    int depth = 0;
                    
                    while(true) {
                        // Safety Bounds (64-bit safe)
                        if(curr >= arena_capacity) break;
                        
                        MCTSNode& node = node_arena[curr];
                        
                        if(node.is_terminal) {
                            float val = node.terminal_value;
                            float curr_val = val; 
                            // Backup (64-bit safe)
                            for(size_t i=path.size(); i-->0; ) {
                                size_t idx = path[i];
                                MCTSNode& n = node_arena[idx];
                                
                                n.visit_count++;
                                float old_v = n.value_sum.load(std::memory_order_relaxed);
                                while(!n.value_sum.compare_exchange_weak(old_v, old_v + curr_val,
                                                                        std::memory_order_release,
                                                                        std::memory_order_relaxed));
                                
                                // Decrease VL (Atomic)
                                n.virtual_loss--;
                                
                                curr_val = -curr_val;
                            }
                            break; 
                        }
                        
                        int expected = 0;
                        if(node.expansion_state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
                            // Expansion Leader
                            // VL is already held from descent.
                            
                            {
                                std::unique_lock<std::mutex> lock(queue_mutex);
                                inference_queue.push({curr, (int)game_idx});  // Cast to int for InferenceRequest
                            }
                            queue_cv.notify_one();
                            
                            // Wait for Expansion
                            while(node.expansion_state.load(std::memory_order_acquire) != 2) {
                                if(stop_signal) return;
                                std::this_thread::yield();
                            }
                            
                            // Backup (64-bit safe)
                            float val = node.terminal_value;
                            float curr_val = val; 
                            for(size_t i=path.size(); i-->0; ) {
                                 size_t idx = path[i];
                                 MCTSNode& n = node_arena[idx];
                                 
                                 n.visit_count++;
                                 float old_v = n.value_sum.load(std::memory_order_relaxed);
                                 while(!n.value_sum.compare_exchange_weak(old_v, old_v + curr_val,
                                                                         std::memory_order_release,
                                                                         std::memory_order_relaxed));
                                 
                                 // Decrease VL
                                 n.virtual_loss--;
                                 
                                 curr_val = -curr_val;
                            }
                            break;
                        } else {
                            // Follower or Already Expanded
                            if(expected == 1) {
                                while(node.expansion_state.load(std::memory_order_acquire) != 2) {
                                    if(stop_signal) return;
                                    std::this_thread::yield();
                                }
                            }
                            
                            // Selection (64-bit safe)
                            // CRITICAL: Must acquire fence to see all child data after expansion
                            std::atomic_thread_fence(std::memory_order_acquire);
                            
                            size_t best_child = SIZE_MAX;  // Invalid marker
                            float best_score = -1e9;
                            int64_t child_start = node.first_child_index;
                            if(child_start < 0) break;  // Not yet set or invalid
                            
                            size_t start = (size_t)child_start;
                            size_t end = start + node.num_children;
                            
                            if(end > arena_capacity) break;
                            
                            for(size_t c=start; c<end; ++c) {
                                float s = ucb(node_arena[c], node);
                                if(s > best_score) {
                                    best_score = s;
                                    best_child = c;
                                }
                            }
                            
                            if(best_child != SIZE_MAX) {
                                curr = best_child;
                                path.push_back(curr);
                                // VL: Add to chosen child (Descent)
                                node_arena[curr].virtual_loss++;
                            } else {
                                 break; 
                            }
                        }
                        
                        depth++;
                        if(depth > 500) break; // Loop breaker
                    } // while tree walk
                } // while sims
            } // for games
        } catch(const std::exception& e) {
            std::cerr << "[W" << worker_id << "] Exception: " << e.what() << std::endl;
        } catch(...) {
            std::cerr << "[W" << worker_id << "] Unknown Exception!" << std::endl;
        }
        
        // Worker Done - CRITICAL SYNCHRONIZATION POINT
        // 1. Memory fence to ensure all node writes are visible before signaling completion
        std::atomic_thread_fence(std::memory_order_release);
        
        // 2. Decrement active workers count
        int active = active_workers.fetch_sub(1, std::memory_order_acq_rel);
        
        // 3. If last worker, signal main thread
        if(active == 1) {
            // Memory fence to ensure main thread sees all updates
            std::atomic_thread_fence(std::memory_order_seq_cst);
            
            std::unique_lock<std::mutex> lock(work_mutex);
            search_in_progress = false;
            main_cv.notify_one();
        }
    }
}

// --- INFERENCER LOOP ---
void MCTS::inferencer_loop() {
    std::vector<InferenceRequest> batch;
    batch.reserve(max_batch_size);
    std::vector<float> encode_buffer(max_batch_size * 14 * 8 * 8); // Pre-allocate max
    
    while(!stop_signal) {
        batch.clear();
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            // Hybrid Wait: 500us timeout to process partial batches
            bool res = queue_cv.wait_for(lock, std::chrono::microseconds(500), [&]{
                return (int)inference_queue.size() >= max_batch_size || stop_signal;
            });
            
            if(stop_signal && inference_queue.empty()) return;
            
            while(!inference_queue.empty() && batch.size() < max_batch_size) {
                batch.push_back(inference_queue.front());
                inference_queue.pop();
            }
        }
        
        if(batch.empty()) continue;
        
        size_t N = batch.size();  // 64-bit batch size
        
        // CRITICAL: Strict bounds checking for MPS slice operations
        if(N <= 0 || N > max_batch_size) {
            std::cerr << "[ERROR] Invalid batch size N=" << N << " (max=" << max_batch_size << ")" << std::endl;
            continue;
        }
        
        try {
            float* encode_ptr = encode_buffer.data();
            
            // Encode with strict bounds checking (64-bit safe)
            for(size_t i=0; i<N; ++i) {
                size_t node_idx = batch[i].node_index;  // CRITICAL: 64-bit!
                if(node_idx >= arena_capacity) {
                    std::cerr << "[ERROR] Invalid node_idx=" << node_idx << " (capacity=" << arena_capacity << ")" << std::endl;
                    continue;
                }
                node_arena[node_idx].state.encode(encode_ptr + i*896);
            }
            
            // Safe GPU buffer resize if needed
            if(gpu_input_buffer.size(0) < (int64_t)N) {
                std::cout << "[MCTS] Resizing GPU buffer from " << gpu_input_buffer.size(0) << " to " << N << std::endl;
                gpu_input_buffer = torch::zeros({static_cast<long long>(N), 14, 8, 8}, torch::kFloat).to(device);
            }
            
            // Safe Slice with explicit N bounds
            auto slice = gpu_input_buffer.slice(0, 0, static_cast<int64_t>(N));
            auto cpu_t = torch::from_blob(encode_buffer.data(), {static_cast<long long>(N), 14, 8, 8}, torch::kFloat);
            slice.copy_(cpu_t, false);
            
            // Inference with detailed error handling
            torch::Tensor p_tens, v_tens;
            static int inference_count = 0;
            ++inference_count;
            try {
                torch::NoGradGuard no_grad;
                auto out = model->forward(slice);
                p_tens = std::get<0>(out).exp().cpu();
                v_tens = std::get<1>(out).cpu();
            } catch(const c10::Error& e) {
                std::cerr << "[FATAL] LibTorch error in inference #" << inference_count << ": " << e.what() << std::endl;
                std::cerr << "[FATAL] Batch size: " << N << ", Arena: " << allocated_count.load() << "/" << arena_capacity << std::endl;
                fflush(stderr);
                torch::mps::synchronize();
                exit(1);
            } catch(const std::exception& e) {
                std::cerr << "[FATAL] Std exception in inference #" << inference_count << ": " << e.what() << std::endl;
                std::cerr << "[FATAL] Batch size: " << N << ", Arena: " << allocated_count.load() << "/" << arena_capacity << std::endl;
                fflush(stderr);
                torch::mps::synchronize();
                exit(1);
            }
            
            float* p_ptr = p_tens.data_ptr<float>();
            float* v_ptr = v_tens.data_ptr<float>();
            
            // Expand (64-bit safe)
            for(size_t i=0; i<N; ++i) {
                size_t node_idx = batch[i].node_index;  // CRITICAL: 64-bit!
                std::vector<float> policy(p_ptr + i*4352, p_ptr + (i+1)*4352);
                float val = v_ptr[i];
                
                // Mirror logic (If Black turn)
                // Note: state was encoded. encoded state is usually canonical (player=white perspective).
                // If turn is Black, we flip policy?
                // MCTSNode state is raw?
                // Helper check:
                if(node_idx < arena_capacity) {
                    if(node_arena[node_idx].state.turn == BLACK) { 
                         std::vector<float> flipped(4352, 0.0f);
                         for(int a=0; a<4352; ++a) {
                             if(MIRROR_TABLE[a] < 4352) flipped[MIRROR_TABLE[a]] = policy[a];
                         }
                         policy = flipped;
                    }
                    expand_node(node_idx, policy, val);
                }
            }
        } catch(const std::exception& e) {
             std::cerr << "[Inferencer] Exception: " << e.what() << std::endl;
             // Unblock waiting workers to avoid deadlock
             for(auto& req : batch) {
                 size_t idx = req.node_index;  // 64-bit!
                 if(idx < arena_capacity) {
                     // Force expand to dummy to unblock
                     node_arena[idx].expansion_state.store(2, std::memory_order_release);
                 }
             }
        }
    }
}

void MCTS::expand_node(size_t node_idx, const std::vector<float>& policy, float value) {
    if(node_idx >= arena_capacity) return;  // size_t is always >= 0
    MCTSNode& node = node_arena[node_idx];
    
    int winner = node.state.get_winner();
    if(winner != 2) {
        node.is_terminal = true;
        node.terminal_value = (winner == 0) ? 0.0f : (winner == node.player ? 1.0f : -1.0f);
        node.expansion_state.store(2, std::memory_order_release);
        return;
    }
    
    auto moves = node.state.get_legal_moves();
    int count = moves.size();
    if(count == 0) {
        node.is_terminal = true;
        node.terminal_value = 0.0f;
        node.expansion_state.store(2, std::memory_order_release);
        return;
    }
    
    size_t start_idx = alloc_nodes(count);
    node.first_child_index = (int64_t)start_idx;
    node.num_children = (uint32_t)count;
    
    float sum = 0.0f;
    for(const auto& m : moves) {
        int a = m.from*64 + m.to;
        if(a < 4352) sum += policy[a];
    }
    
    // Correct loop (64-bit safe)
    for(size_t i=0; i<(size_t)count; ++i) {
        Move m = moves[i];
        int a = m.from*64 + m.to;
         float p = (sum > 0 && a < 4352) ? (policy[a]/sum) : (1.0f/count);
         Chess next = node.state;
         next.push(m);
         
         new (&node_arena[start_idx+i]) MCTSNode();
         node_arena[start_idx+i].init(next, -node.player, (int64_t)node_idx, a, p);
    }
    
    node.terminal_value = value;
    
    // CRITICAL MEMORY BARRIER: Ensure all child node writes are visible
    // Before expansion_state=2. Without this fence, other threads may see
    // expansion_state=2 but read stale first_child_index or uninitialized children
    std::atomic_thread_fence(std::memory_order_release);
    
    node.expansion_state.store(2, std::memory_order_release);
}

float MCTS::ucb(const MCTSNode& node, const MCTSNode& parent) const {
    // CRITICAL ACQUIRE FENCE: Ensure we read consistent atomic values
    // Without this, we may see partial updates during backup
    std::atomic_thread_fence(std::memory_order_acquire);
    
    // Read atomics with explicit acquire ordering for consistency
    int vc = node.visit_count.load(std::memory_order_acquire);
    int vl = node.virtual_loss.load(std::memory_order_acquire);
    float vs = node.value_sum.load(std::memory_order_acquire);
    int parent_vc = parent.visit_count.load(std::memory_order_acquire);
    
    int total = vc + vl;
    float node_val = 0.0f;
    if(total > 0) {
        node_val = (vs - vl) / total;
    } else {
        node_val = -0.2f;
    }
    float q = -node_val;
    float u = c_puct * node.prior * std::sqrt((float)parent_vc) / (1.0f + total);
    return q + u;
}

std::vector<std::vector<float>> MCTS::search_async(const std::vector<size_t>& root_indices) {
    if(allocated_count == 0) {
        // First run or reset. Check arena?
        // Allocator is fine.
    }
    
    size_t N = root_indices.size();
    std::vector<std::atomic<int>> sims_remaining(N);
    for(size_t i=0; i<N; ++i) sims_remaining[i] = num_simulations;
    
    // Set Job State
    {
        std::lock_guard<std::mutex> lock(work_mutex);
        current_roots = &root_indices;
        current_sims_remaining = sims_remaining.data();
        active_workers = worker_threads.size();
        search_in_progress = true;
    }
    
    // Wake Workers
    work_cv.notify_all();
    
    // Wait for completion
    {
        std::unique_lock<std::mutex> lock(work_mutex);
        main_cv.wait(lock, [&]{ return !search_in_progress; });
    }
    
    // CRITICAL: Acquire fence to ensure all worker writes are visible before reading results
    std::atomic_thread_fence(std::memory_order_acquire);
    
    // std::cout << "[MCTS] Search Finished." << std::endl; // Optional debug
    
    // Aggregate Results (64-bit safe)
    std::vector<std::vector<float>> results;
    for(size_t root_idx : root_indices) {
        MCTSNode& root = node_arena[root_idx];
        std::vector<float> pi(4352, 0.0f);
        float sum = 0.0f;
        int64_t start = root.first_child_index;
        size_t end = (size_t)start + root.num_children;
        if(start >= 0) {
            for(size_t c=(size_t)start; c<end; ++c) {
                 MCTSNode& child = node_arena[c];
                 int action = child.move_from_parent;
                 if(action >= 0 && action < 4352) {
                     pi[action] = (float)child.visit_count;
                     sum += pi[action];
                 }
            }
        }
        if(sum > 0) { for(float& p : pi) p /= sum; }
        results.push_back(pi);
    }
    return results;
}

int MCTS::select_action(size_t root_idx, float temp) {
    MCTSNode& root = node_arena[root_idx];
    std::vector<float> pi(4352, 0.0f);
    float sum = 0.0f;
    int64_t start = root.first_child_index;
    size_t end = (start >= 0) ? (size_t)start + root.num_children : 0;
    if(start >= 0) {
        for(size_t c=(size_t)start; c<end; ++c) {
            if(node_arena[c].visit_count > 0) {
                 int a = node_arena[c].move_from_parent;
                 pi[a] = node_arena[c].visit_count;
                 sum += pi[a];
            }
        }
    }
    std::vector<float> probs = pi; 
    if (temp == 0) {
        auto argmax = std::max_element(probs.begin(), probs.end());
        return std::distance(probs.begin(), argmax);
    }
    float psum = 0.0f;
    for(float& p : probs) {
        p = std::pow(p, 1.0f/temp);
        psum += p;
    }
    if(psum > 0) { for(float& p : probs) p /= psum; } 
    else { return root.first_child_index != -1 ? node_arena[root.first_child_index].move_from_parent : 0; }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(probs.begin(), probs.end());
    return d(gen);
}
