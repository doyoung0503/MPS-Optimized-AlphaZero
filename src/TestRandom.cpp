#include <iostream>
#include <memory>
#include <filesystem>
#include <random>
#include "Network.h"
#include "Chess.h"
#include "MCTS.h"

int play_match(std::shared_ptr<AlphaZeroNet> model, int sims, torch::Device device, int games = 20) {
    std::cout << "Playing " << games << " games against Random..." << std::endl;
    
    MCTS mcts(model, sims, 1.5f, device);
    
    int wins = 0;
    int draws = 0;
    int losses = 0;
    
    // Random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for(int i=0; i<games; ++i) {
        Chess state;
        // Alternate colors
        int ai_player = (i % 2 == 0) ? 1 : -1; // 1=White, -1=Black
        
        // Loop
        while(state.get_winner() == 2) {
            int current_player = state.turn;
            Move best_move;
            
            if(current_player == ai_player) {
                // AI Turn
                // Single game search
                std::vector<Chess> batch = {state};
                auto policies = mcts.search_batch(batch);
                
                // MPS sync before accessing results
                if(device == torch::kMPS) {
                    torch::mps::synchronize();
                }
                
                int action = mcts.select_action(policies[0], 0.1f); // Low temp for play
                
                // Decode
                best_move.from = action / 64;
                best_move.to = action % 64;
                best_move.promotion = 0;
                
                // Promotion check (simple hack same as Trainer)
                int p_type = abs(state.board[best_move.from]);
                if(p_type == 1) {
                    int r = best_move.to / 8;
                    if(r == 0 || r == 7) best_move.promotion = 5;
                }
            } else {
                // Random Turn
                auto moves = state.get_legal_moves();
                if(moves.empty()) break;
                std::uniform_int_distribution<> d(0, moves.size()-1);
                best_move = moves[d(gen)];
            }
            
            state.push(best_move);
            
            // Check max moves
            if(state.fullmoves > 150) break; // Draw
        }
        
        int winner = state.get_winner();
        if(winner == 2) winner = 0; // Draw by move limit
        
        std::cout << "Game " << (i+1) << ": ";
        if(winner == ai_player) { std::cout << "AI Won"; wins++; }
        else if(winner == 0) { std::cout << "Draw"; draws++; }
        else { std::cout << "Random Won"; losses++; }
        std::cout << " (AI Color: " << (ai_player==1?"White":"Black") << ")" << std::endl;
    }
    
    std::cout << "\nResults: " << wins << "W - " << draws << "D - " << losses << "L" << std::endl;
    float win_rate = 100.0f * wins / games;
    std::cout << "Win Rate: " << win_rate << "%" << std::endl;
    return wins;
}

int main(int argc, char* argv[]) {
    if(argc < 2) {
        std::cout << "Usage: ./test_random <checkpoint_path>" << std::endl;
        return 1;
    }
    
    std::string path = argv[1];
    
    // Model
    auto model = std::make_shared<AlphaZeroNet>(14, 8, 4352, 10, 128);
    torch::Device device = torch::kCPU;
    if (torch::mps::is_available()) device = torch::kMPS;
    else if (torch::cuda::is_available()) device = torch::kCUDA;
    
    model->to(device);
    
    // MPS sync after model transfer
    if(device == torch::kMPS) {
        torch::mps::synchronize();
    }
    
    if (std::filesystem::exists(path)) {
        torch::load(model, path);
        std::cout << "Loaded: " << path << std::endl;
        model->eval();
        
        // MPS sync after loading weights
        if(device == torch::kMPS) {
            torch::mps::synchronize();
        }
        
        play_match(model, 50, device, 10); // 50 sims, 10 games for quick test
    } else {
        std::cout << "File not found: " << path << std::endl;
    }
    return 0;
}

