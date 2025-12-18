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
    // Size: 200 sims * 32 games = ~6.4k nodes per search.
    // Full game (150 moves) * 6.4k = ~1M nodes. 
    // 3M nodes (~1.1GB) provides safety margin.
    int num_nodes = 3000000; 
    
    size_t size_bytes = (size_t)num_nodes * sizeof(MCTSNode);
    int prot = PROT_READ | PROT_WRITE;
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    
    node_arena = (MCTSNode*)mmap(NULL, size_bytes, prot, flags, -1, 0);
    if(node_arena == MAP_FAILED) {
        perror("mmap");
        std::cerr << "Failed to allocate Arena of size " << size_bytes << std::endl;
        exit(1);
    }
    
    arena_capacity = num_nodes;
    allocated_count = 0;
    
    std::cout << "[MCTS] Arena Allocated: " << (size_bytes / 1024 / 1024) << " MB (" << num_nodes << " nodes)" << std::endl;
    
    // 4. Alignment Check
    uintptr_t addr = reinterpret_cast<uintptr_t>(node_arena);
    if(addr % 64 != 0) {
        std::cerr << "[CRITICAL] Arena not aligned to 64 bytes! Addr: " << addr << std::endl;
        exit(1);
    } else {
        std::cout << "[MCTS] Arena Aligned (64 bytes). Addr: " << addr << std::endl;
    }
    
    // 2. Persistent GPU Buffer
    gpu_input_buffer = torch::zeros({256, 14, 8, 8}, torch::kFloat).to(device);
    
    // 3. Start Threads
    int hw_concurrency = std::thread::hardware_concurrency();
    int num_workers = std::max(4, hw_concurrency); 
    // Use full concurrency for workers, inference runs on separate thread (hyperthreading or efficient scheduling)
    // Or reserved 1? Let's use hw_concurrency.
    
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

int MCTS::create_root(const Chess& state) {
    int idx = alloc_nodes(1);
    // Placement new to construct atomics properly
    new (&node_arena[idx]) MCTSNode(); 
    node_arena[idx].init(state, state.turn, -1, -1, 0.0f);
    
    // Add VL for root? No, worker loop adds it.
    
    return idx;
}

int MCTS::alloc_nodes(int count) {
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
    
    return (int)idx;
}

MCTSNode& MCTS::get_node(int index) {
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
        
        // Job Received
        const std::vector<int>& roots = *current_roots;
        int num_games = roots.size();
        
        try {
            for(int game_idx = worker_id; game_idx < num_games; game_idx += stride) {
                int root_idx = roots[game_idx];
                if(root_idx < 0 || root_idx >= arena_capacity) continue;
                
                std::atomic<int>& remaining = current_sims_remaining[game_idx];
                
                while(remaining > 0) {
                    remaining--;
                    
                    int curr = root_idx;
                    // VL: Add to Root immediately (Descent phase)
                    node_arena[curr].virtual_loss++;
                    
                    std::vector<int> path; 
                    path.reserve(64);
                    path.push_back(curr);
                    
                    int depth = 0;
                    
                    while(true) {
                        // Safety Bounds
                        if(curr < 0 || curr >= arena_capacity) break;
                        
                        MCTSNode& node = node_arena[curr];
                        
                        if(node.is_terminal) {
                            float val = node.terminal_value;
                            float curr_val = val; 
                            // Backup
                            for(int i=path.size()-1; i>=0; --i) {
                                int idx = path[i];
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
                                inference_queue.push({curr, game_idx});
                            }
                            queue_cv.notify_one();
                            
                            // Wait for Expansion
                            while(node.expansion_state.load(std::memory_order_acquire) != 2) {
                                if(stop_signal) return;
                                std::this_thread::yield();
                            }
                            
                            // Backup
                            float val = node.terminal_value;
                            float curr_val = val; 
                            for(int i=path.size()-1; i>=0; --i) {
                                 int idx = path[i];
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
                            
                            // Selection
                            int best_child = -1;
                            float best_score = -1e9;
                            int start = node.first_child_index;
                            int end = start + node.num_children;
                            
                            if(start < 0 || end > arena_capacity) break;
                            
                            for(int c=start; c<end; ++c) {
                                float s = ucb(node_arena[c], node);
                                if(s > best_score) {
                                    best_score = s;
                                    best_child = c;
                                }
                            }
                            
                            if(best_child != -1) {
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
        
        // Worker Done
        int active = active_workers.fetch_sub(1);
        if(active == 1) { // Last one out
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
        
        int N = batch.size();
        
        try {
            float* encode_ptr = encode_buffer.data();
            
            // Encode
            for(int i=0; i<N; ++i) {
                int node_idx = batch[i].node_index;
                if(node_idx >= 0 && node_idx < arena_capacity) {
                    node_arena[node_idx].state.encode(encode_ptr + i*896);
                }
            }
            
            if(gpu_input_buffer.size(0) < N) {
                 // Should not happen if max_batch_size is constant and buffer init is correct
                 gpu_input_buffer = torch::zeros({N, 14, 8, 8}, torch::kFloat).to(device);
            }
            
            // Safe Slice
            auto slice = gpu_input_buffer.slice(0, 0, N);
            auto cpu_t = torch::from_blob(encode_buffer.data(), {N, 14, 8, 8}, torch::kFloat);
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
                std::cerr << "[FATAL] Batch size was: " << N << std::endl;
                exit(1);
            } catch(const std::exception& e) {
                std::cerr << "[FATAL] Std exception in inference #" << inference_count << ": " << e.what() << std::endl;
                exit(1);
            }
            
            float* p_ptr = p_tens.data_ptr<float>();
            float* v_ptr = v_tens.data_ptr<float>();
            
            // Expand
            for(int i=0; i<N; ++i) {
                int node_idx = batch[i].node_index;
                std::vector<float> policy(p_ptr + i*4352, p_ptr + (i+1)*4352);
                float val = v_ptr[i];
                
                // Mirror logic (If Black turn)
                // Note: state was encoded. encoded state is usually canonical (player=white perspective).
                // If turn is Black, we flip policy?
                // MCTSNode state is raw?
                // Helper check:
                if(node_idx >= 0 && node_idx < arena_capacity) {
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
                 int idx = req.node_index;
                 if(idx >= 0 && idx < arena_capacity) {
                     // Force expand to dummy to unblock
                     node_arena[idx].expansion_state.store(2, std::memory_order_release);
                 }
             }
        }
    }
}

void MCTS::expand_node(int node_idx, const std::vector<float>& policy, float value) {
    if(node_idx < 0 || node_idx >= arena_capacity) return;
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
    
    int start_idx = alloc_nodes(count);
    node.first_child_index = start_idx;
    node.num_children = (uint16_t)count;
    
    float sum = 0.0f;
    for(const auto& m : moves) {
        int a = m.from*64 + m.to;
        if(a < 4352) sum += policy[a];
    }
    
    // Correct loop
    for(int i=0; i<count; ++i) {
        Move m = moves[i];
        int a = m.from*64 + m.to;
         float p = (sum > 0 && a < 4352) ? (policy[a]/sum) : (1.0f/count);
         Chess next = node.state;
         next.push(m);
         
         new (&node_arena[start_idx+i]) MCTSNode();
         node_arena[start_idx+i].init(next, -node.player, node_idx, a, p);
    }
    
    node.terminal_value = value;
    node.expansion_state.store(2, std::memory_order_release);
}

float MCTS::ucb(const MCTSNode& node, const MCTSNode& parent) const {
    int total = node.visit_count + node.virtual_loss;
    float node_val = 0.0f;
    if(total > 0) {
        node_val = (node.value_sum - node.virtual_loss) / total;
    } else {
        node_val = -0.2f;
    }
    float q = -node_val;
    float u = c_puct * node.prior * std::sqrt((float)parent.visit_count) / (1.0f + total);
    return q + u;
}

std::vector<std::vector<float>> MCTS::search_async(const std::vector<int>& root_indices) {
    if(allocated_count == 0) {
        // First run or reset. Check arena?
        // Allocator is fine.
    }
    
    int N = root_indices.size();
    std::vector<std::atomic<int>> sims_remaining(N);
    for(int i=0; i<N; ++i) sims_remaining[i] = num_simulations;
    
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
    
    // std::cout << "[MCTS] Search Finished." << std::endl; // Optional debug
    
    // Aggregate Results
    std::vector<std::vector<float>> results;
    for(int root_idx : root_indices) {
        MCTSNode& root = node_arena[root_idx];
        std::vector<float> pi(4352, 0.0f);
        float sum = 0.0f;
        int start = root.first_child_index;
        int end = start + root.num_children;
        for(int c=start; c<end; ++c) {
             MCTSNode& child = node_arena[c];
             int action = child.move_from_parent;
             if(action >= 0 && action < 4352) {
                 pi[action] = (float)child.visit_count;
                 sum += pi[action];
             }
        }
        if(sum > 0) { for(float& p : pi) p /= sum; }
        results.push_back(pi);
    }
    return results;
}

int MCTS::select_action(int root_idx, float temp) {
    MCTSNode& root = node_arena[root_idx];
    std::vector<float> pi(4352, 0.0f);
    float sum = 0.0f;
    int start = root.first_child_index;
    int end = start + root.num_children;
    for(int c=start; c<end; ++c) {
        if(node_arena[c].visit_count > 0) {
             int a = node_arena[c].move_from_parent;
             pi[a] = node_arena[c].visit_count;
             sum += pi[a];
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
