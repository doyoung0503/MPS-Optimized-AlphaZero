#include <iostream>
#include <vector>
#include <chrono>
#include <memory>
#include "Network.h"
#include "Chess.h"
#include "MCTS.h"

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "🚀 AlphaZero Batch Size Grid Search" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Device
    auto device = torch::kCPU;
    if (torch::mps::is_available()) device = torch::kMPS;
    else if (torch::cuda::is_available()) device = torch::kCUDA;
    
    // Model
    auto model = std::make_shared<AlphaZeroNet>(14, 8, 4352, 10, 128);
    model->to(device);
    
    // Test Batches
    std::vector<int> batches = {32, 64, 128, 256, 512, 1024};
    int simulations = 100;
    
    std::cout << "Simulations per search: " << simulations << "\n" << std::endl;
    std::cout << "| Batch Size | Time (s) | Speed (Games/s) | Speed (Sims/s) |" << std::endl;
    std::cout << "|------------|----------|-----------------|----------------|" << std::endl;
    
    for (int batch_size : batches) {
        // Prepare Games
        std::vector<Chess> games(batch_size); 
        MCTS mcts(model, simulations, 1.5f, device);
        
        // Warmup (MPS lazy initialization)
        mcts.search_batch(games);
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        
        // Run 5 searches to average noise
        for(int k=0; k<5; ++k) mcts.search_batch(games);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        
        double avg_time = diff.count() / 5.0;
        double games_per_sec = batch_size / avg_time;
        double sims_per_sec = (batch_size * simulations) / avg_time;
        
        printf("| %10d | %8.4f | %15.2f | %14.0f |\n", batch_size, avg_time, games_per_sec, sims_per_sec);
        std::cout << std::flush;
    }
    
    std::cout << "\nDone!" << std::endl;
    return 0;
}
