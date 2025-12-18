#pragma once
#include <torch/torch.h>

struct ResidualBlock : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};

    ResidualBlock(int filters) {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(filters, filters, 3).padding(1).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(filters));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(filters, filters, 3).padding(1).bias(false)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(filters));
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor residual = x;
        x = torch::relu(bn1(conv1(x)));
        x = bn2(conv2(x));
        x += residual;
        return torch::relu(x);
    }
};

struct AlphaZeroNet : torch::nn::Module {
    torch::nn::Conv2d conv_input{nullptr};
    torch::nn::BatchNorm2d bn_input{nullptr};
    std::vector<std::shared_ptr<ResidualBlock>> res_blocks;
    
    // Policy head
    torch::nn::Conv2d policy_conv{nullptr};
    torch::nn::BatchNorm2d policy_bn{nullptr};
    torch::nn::Linear policy_fc{nullptr};
    
    // Value head
    torch::nn::Conv2d value_conv{nullptr};
    torch::nn::BatchNorm2d value_bn{nullptr};
    torch::nn::Linear value_fc1{nullptr}, value_fc2{nullptr};

    AlphaZeroNet(int in_channels, int board_size, int action_size, int num_res_blocks, int filters) {
        // Input
        conv_input = register_module("conv_input", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, filters, 3).padding(1).bias(false)));
        bn_input = register_module("bn_input", torch::nn::BatchNorm2d(filters));
        
        // Residual Tower
        for (int i = 0; i < num_res_blocks; ++i) {
            auto block = std::make_shared<ResidualBlock>(filters);
            res_blocks.push_back(block);
            register_module("res_block_" + std::to_string(i), block);
        }
        
        // Policy Head
        policy_conv = register_module("policy_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(filters, 32, 1).bias(false)));
        policy_bn = register_module("policy_bn", torch::nn::BatchNorm2d(32));
        policy_fc = register_module("policy_fc", torch::nn::Linear(32 * board_size * board_size, action_size));
        
        // Value Head
        value_conv = register_module("value_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(filters, 1, 1).bias(false)));
        value_bn = register_module("value_bn", torch::nn::BatchNorm2d(1));
        value_fc1 = register_module("value_fc1", torch::nn::Linear(1 * board_size * board_size, 256));
        value_fc2 = register_module("value_fc2", torch::nn::Linear(256, 1));
        
        // === Xavier Uniform Initialization for output layers ===
        // Prevents output variance explosion in large action space
        torch::nn::init::xavier_uniform_(policy_fc->weight);
        torch::nn::init::zeros_(policy_fc->bias);
        torch::nn::init::xavier_uniform_(value_fc2->weight);
        torch::nn::init::zeros_(value_fc2->bias);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        x = torch::relu(bn_input(conv_input(x)));
        
        for (auto& block : res_blocks) {
            x = block->forward(x);
        }
        
        // Policy head
        torch::Tensor p = torch::relu(policy_bn(policy_conv(x)));
        p = p.flatten(1);
        p = policy_fc(p);
        
        // === Temperature + Logit Clamping ===
        // Temperature: softens distribution, prevents overconfident predictions
        p = p / 1.5f;  // Temperature = 1.5
        
        // Tighter clamping prevents extreme logits
        p = torch::clamp(p, -10.0f, 10.0f);
        
        p = torch::log_softmax(p, 1);
        
        // Value head
        torch::Tensor v = torch::relu(value_bn(value_conv(x)));
        v = v.flatten(1);
        v = torch::relu(value_fc1(v));
        v = torch::tanh(value_fc2(v));
        
        return {p, v};
    }
};

