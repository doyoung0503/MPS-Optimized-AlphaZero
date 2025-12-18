#include <iostream>
#include <memory>
#include <chrono>
#include <filesystem>
#include "Network.h"
#include "Chess.h"
#include "MCTS.h"
#include "Trainer.h"

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "  8x8 Chess AlphaZero (C++ / LibTorch)    " << std::endl;
    std::cout << "==========================================" << std::endl;

    // Device
    auto device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "Device: CUDA" << std::endl;
        device = torch::kCUDA;
    } else if (torch::mps::is_available()) {
        std::cout << "Device: MPS (Apple Silicon)" << std::endl;
        device = torch::kMPS;
    } else {
        std::cout << "Device: CPU" << std::endl;
    }

    // Model
    std::cout << "Initializing Model (14 channels, 128 filters, 10 blocks)..." << std::endl;
    auto model = std::make_shared<AlphaZeroNet>(14, 8, 4352, 10, 128);
    model->to(device);
    
    // Trainer
    Trainer trainer(model, device);
    
    int ITERATIONS = 30;  // Extended training
    int GAMES_PER_ITER = 64; // M4 Pro can handle 64 concurrent games with 60M arena
    int MAX_SIMS = 200;  // Higher simulations
    int TRAIN_BATCHES = 30;  // More batches
    
    // Auto-Resume Logic
    std::cout << "Current path: " << std::filesystem::current_path() << std::endl;
    int start_iter = 1;
    for(int i=100; i>=1; --i) {
        std::string path = "checkpoint_" + std::to_string(i) + ".pt";
        if(std::filesystem::exists(path)) {
            std::cout << "Found checkpoint: " << path << std::endl;
            trainer.load_checkpoint(path);
            start_iter = i + 1;
            break;
        }
    }

    std::cout << "\nStarting Training..." << std::endl;
    std::cout << "Iterations: " << ITERATIONS << std::endl;
    std::cout << "Games/Iter: " << GAMES_PER_ITER << std::endl;
    std::cout << "Simulations (Max): " << MAX_SIMS << std::endl;
    std::cout << "Start Iter: " << start_iter << std::endl;
    
    trainer.train(ITERATIONS, GAMES_PER_ITER, MAX_SIMS, TRAIN_BATCHES, start_iter);
    
    std::cout << "\nDone!" << std::endl;
    return 0;
}
