#include "Trainer.h"
#include "AlignedBuffer.h"
#include <iostream>
#include <filesystem>
#include <random>
#import <Foundation/Foundation.h>

namespace fs = std::filesystem;

Trainer::Trainer(std::shared_ptr<AlphaZeroNet> net, torch::Device dev)
    : model(net), device(dev), 
      optimizer(net->parameters(), torch::optim::AdamOptions(0.0002)) {} 

void Trainer::train(int iterations, int games_per_iter, int max_sims, int train_batches, int start_iter) {
    std::cout << "Starting C++ Training Loop (Async Pool Strategy)..." << std::endl;
    std::cout << "Resuming from Iteration " << start_iter << std::endl;
    
    for (int iter = start_iter; iter <= iterations; ++iter) {
        int current_sims = 50;
        if (iter > 5) current_sims = 100;
        if (iter > 10) current_sims = 200;
        if (iter > 20) current_sims = max_sims;
        
        std::cout << "\n---------------------------------" << std::endl;
        std::cout << "Iteration " << iter << "/" << iterations << " [Sims: " << current_sims << "]" << std::endl;
        std::cout << "---------------------------------" << std::endl;
        
        try {
            auto start = std::chrono::high_resolution_clock::now();
            self_play(games_per_iter, current_sims);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::cout << "Self-play complete in " << diff.count() << "s (" << (diff.count()/games_per_iter) << "s/game)" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Exception in self_play: " << e.what() << std::endl;
            throw; 
        }

        std::cout << "Buffer size: " << buffer.size() << std::endl;
        
        try {
            training_step(train_batches);
            if(device.is_mps()) torch::mps::synchronize();
        } catch (const std::exception& e) {
            std::cerr << "Exception in training_step: " << e.what() << std::endl;
            throw;
        }
        
        std::string path = "checkpoint_" + std::to_string(iter) + ".pt";
        save_checkpoint(path);
    }
}

void Trainer::self_play(int num_games, int simulations) {
    std::cout << "[Self-Play] Playing " << num_games << " games..." << std::endl;
    model->eval(); 
    // Batch Size 64 for 64 games? Or 32? Let's use 64.
    // Batch Size 64 for 64 games? Or 32? Let's use 64.
    MCTS mcts(model, simulations, 1.5f, 64, device);
    mcts.reset_pool(); // Reset arena cursor
    
    std::vector<Chess> games(num_games);
    std::vector<bool> finished(num_games, false);
    
    struct GameStep {
        Chess state_copy;
        std::vector<float> policy;
        int player;
        float value = 0.0f;
    };
    std::vector<std::vector<GameStep>> histories(num_games);
    for(auto& h : histories) h.reserve(256); 
    
    int total_moves = 0;
    int finished_games = 0;
    auto last_print = std::chrono::steady_clock::now();
    
    // Roots are INDICES now (64-bit for >2GB arena)
    std::vector<size_t> root_indices;
    root_indices.reserve(num_games);
    for(auto& g : games) {
        root_indices.push_back(mcts.create_root(g));
    }
    
    while(finished_games < num_games) {
        // 1. Prepare Active Roots (64-bit indices)
        std::vector<size_t> active_root_indices;
        std::vector<int> active_game_indices;  // Game indices can stay int
        active_root_indices.reserve(num_games);
        active_game_indices.reserve(num_games);
        
        for(int i=0; i<num_games; ++i) {
            if(!finished[i]) {
                active_root_indices.push_back(root_indices[i]);
                active_game_indices.push_back(i);
            }
        }
        
        // 2. Async Search
        auto policies = mcts.search_async(active_root_indices);
        
        // 3. Process Results
        for(int k=0; k<active_game_indices.size(); ++k) {
            int i = active_game_indices[k];
            int root_idx = active_root_indices[k];
            
            // Stats
            int game_len = histories[i].size();
            float temp = (game_len < 30) ? 1.0f : 0.05f; // Lower temp later
            
            int action = mcts.select_action(root_idx, temp);
            
            // Advance Game State using Tree (if child exists)
            MCTSNode& current_root = mcts.get_node(root_idx);
            int next_root_idx = -1;
            
            if(current_root.expansion_state.load(std::memory_order_acquire) == 2) {
                int start = current_root.first_child_index;
                int end = start + current_root.num_children;
                for(int c=start; c<end; ++c) {
                    if(mcts.get_node(c).move_from_parent == action) {
                        next_root_idx = c;
                        break;
                    }
                }
            }
            
            // Apply Action
            if(next_root_idx != -1) {
                games[i] = mcts.get_node(next_root_idx).state;
                root_indices[i] = next_root_idx;
                
                // Store History
                histories[i].push_back({mcts.get_node(root_idx).state, policies[k], mcts.get_node(root_idx).state.turn}); 
            } else {
                // Should not happen if select_action returns valid move from child
                // But if temp > 0, it might pick a move that was legal but maybe node not fully expanded?
                // No, select_action picks from Children.
                // So Child MUST exist.
                // The only case is if select_action returned something else?
                // Logic seems safe. Fallback just in case.
                 auto moves = games[i].get_legal_moves();
                 for(const auto& m : moves) {
                     int a = m.from*64 + m.to;
                     if(a == action) {
                         games[i].push(m);
                         break;
                     }
                 }
                 root_indices[i] = mcts.create_root(games[i]);
                 histories[i].push_back({mcts.get_node(root_idx).state, policies[k], mcts.get_node(root_idx).state.turn});
            }
            
            
            total_moves++;
            
            // === MEMORY CORRUPTION DETECTION ===
            // If moves_count jumps unexpectedly, stack/heap has been corrupted
            static int last_total_moves = 0;
            int delta = total_moves - last_total_moves;
            if(delta > 1000 || delta < 0) {
                std::cerr << "[FATAL] Memory corruption detected!" << std::endl;
                std::cerr << "  total_moves jumped from " << last_total_moves << " to " << total_moves << std::endl;
                std::cerr << "  delta = " << delta << " (expected 1)" << std::endl;
                std::cerr << "  Terminating to prevent further corruption." << std::endl;
                exit(1);
            }
            last_total_moves = total_moves;
            
            if(games[i].is_game_over()) {
                finished[i] = true;
                finished_games++;
                int winner = games[i].get_winner();
                // Assign rewards...
            }
        }
        
        auto now = std::chrono::steady_clock::now();
        if(std::chrono::duration_cast<std::chrono::seconds>(now - last_print).count() >= 1) {
             std::cout << "[Self-Play] Active: " << (num_games - finished_games) 
                       << " | Finished: " << finished_games 
                       << " | Moves: " << total_moves << std::endl;
             last_print = now;
        }
    }
    std::cout << std::endl;
    while(buffer.size() > buffer_size) buffer.pop_front();
}
// ... (rest of Trainer methods same) ...
void Trainer::training_step(int num_batches) {
    if(buffer.size() < batch_size) return;
    std::cout << "[Training] Running " << num_batches << " batches..." << std::endl;
    model->train();
    float total_loss = 0.0f;
    for(int b=0; b<num_batches; ++b) {
        std::vector<torch::Tensor> states;
        std::vector<torch::Tensor> target_pis;
        std::vector<torch::Tensor> target_vs;
        for(int k=0; k<batch_size; ++k) {
            int idx = rand() % buffer.size();
            auto& ex = buffer[idx];
            auto canonical = ex.state.get_canonical();
            auto enc = canonical.encode();
            states.push_back(torch::from_blob(enc.data(), {14, 8, 8}, torch::kFloat).clone());
            std::vector<float> policy_vec(ex.policy.begin(), ex.policy.end());
            float policy_sum = 0.0f;
            for(float p : policy_vec) policy_sum += p;
            if(policy_sum < 1e-8) { for(float& p : policy_vec) p = 1.0f / 4352.0f; } 
            else { for(float& p : policy_vec) p /= policy_sum; }
            auto policy_tensor = torch::from_blob(policy_vec.data(), {4352}, torch::kFloat).clone();
            target_pis.push_back(policy_tensor);
            target_vs.push_back(torch::tensor({ex.value}));
        }
        auto batch_states = torch::stack(states).contiguous().to(device);
        auto batch_pis = torch::stack(target_pis).contiguous().to(device);
        auto batch_vs = torch::stack(target_vs).contiguous().to(device);
        torch::Tensor log_pis;
        torch::Tensor vs;
        @autoreleasepool {
            optimizer.zero_grad();
            auto out = model->forward(batch_states);
            if(device.is_mps()) torch::mps::synchronize();
            log_pis = std::get<0>(out).clone();
            vs = std::get<1>(out).clone();
        } 
        if(device.is_mps()) torch::mps::synchronize();
        float epsilon = 0.1f;
        int num_actions = 4352;
        auto smoothed_target = batch_pis * (1.0f - epsilon) + (epsilon / (float)num_actions);
        if(torch::isnan(smoothed_target).any().item<bool>()) continue;
        auto p_loss = -torch::sum(smoothed_target * log_pis) / batch_size;
        auto v_loss = torch::mse_loss(vs, batch_vs);
        auto loss = p_loss + v_loss;
        if(device.is_mps()) torch::mps::synchronize();
        float loss_val = loss.item<float>();
        if(std::isnan(loss_val) || std::isinf(loss_val)) {
             if(!last_good_checkpoint.empty() && fs::exists(last_good_checkpoint)) {
                load_checkpoint(last_good_checkpoint);
                optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(0.0002));
            }
            return; 
        }
        loss.backward();
        torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
        if(check_weights_nan()) return;
        optimizer.step();
        if(check_weights_nan()) return;
        if(device.is_mps()) torch::mps::synchronize();
        total_loss += loss_val;
    }
    std::cout << "Avg Loss: " << total_loss / num_batches << std::endl;
}

void Trainer::save_checkpoint(const std::string& path) {
    torch::save(model, path);
    last_good_checkpoint = path; 
    std::cout << "Saved checkpoint: " << path << std::endl;
}
void Trainer::load_checkpoint(const std::string& path) {
    if (fs::exists(path)) {
        torch::load(model, path);
        std::cout << "Loaded checkpoint: " << path << std::endl;
    } else {
        std::cout << "Checkpoint not found: " << path << std::endl;
    }
}
bool Trainer::check_weights_nan() {
    for(const auto& param : model->parameters()) {
        if(torch::isnan(param).any().item<bool>()) return true;
        if(param.grad().defined() && torch::isnan(param.grad()).any().item<bool>()) return true;
    }
    return false;
}
