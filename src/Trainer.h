#pragma once
#include "Network.h"
#include "Chess.h"
#include "MCTS.h"
#include <deque>
#include <string>

struct TrainingExample {
    Chess state;
    std::vector<float> policy;
    float value; // Outcome
};

class Trainer {
public:
    Trainer(std::shared_ptr<AlphaZeroNet> net, torch::Device dev);
    
    void train(int iterations, int games_per_iter, int max_sims, int train_batches, int start_iter = 1);
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);

private:
    std::shared_ptr<AlphaZeroNet> model;
    torch::Device device;
    torch::optim::Adam optimizer;
    
    std::deque<TrainingExample> buffer;
    int buffer_size = 30000;
    int batch_size = 64;
    
    // NaN Recovery
    std::string last_good_checkpoint;
    
    void self_play(int num_games, int simulations);
    void training_step(int num_batches);
    bool check_weights_nan();  // Check if any weights are NaN
};
