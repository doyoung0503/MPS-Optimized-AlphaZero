# MPS-Optimized AlphaZero

A high-performance AlphaZero implementation in C++ with LibTorch, optimized for Apple Silicon (MPS backend).

## Features

- **8x8 Chess Engine**: Full chess implementation with legal move generation
- **Deep Neural Network**: ResNet-style policy/value network (10 blocks, 128 filters)
- **Parallel MCTS**: Lock-free Monte Carlo Tree Search with persistent thread pool
- **MPS Optimization**: Native Apple Silicon GPU acceleration via Metal Performance Shaders
- **Arena Allocator**: Custom mmap-based memory management for 30M+ nodes
- **Async Inference**: Producer-consumer batch inference server with 500μs latency timeout

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Trainer                                  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Self-Play                             ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      ││
│  │  │   Game 1    │  │   Game 2    │  │   Game N    │      ││
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      ││
│  │         │                │                │              ││
│  │         └────────────────┼────────────────┘              ││
│  │                          │                               ││
│  │                   ┌──────▼──────┐                        ││
│  │                   │    MCTS     │                        ││
│  │                   │  (Parallel) │                        ││
│  │                   └──────┬──────┘                        ││
│  │                          │                               ││
│  │         ┌────────────────┼────────────────┐              ││
│  │         │                │                │              ││
│  │  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐      ││
│  │  │  Worker 1   │  │  Worker 2   │  │  Worker N   │      ││
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      ││
│  │         └────────────────┼────────────────┘              ││
│  │                          │                               ││
│  │                   ┌──────▼──────┐                        ││
│  │                   │  Inference  │                        ││
│  │                   │   Server    │                        ││
│  │                   │   (MPS)     │                        ││
│  │                   └─────────────┘                        ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Requirements

- macOS 12.0+ (Monterey or later)
- Apple Silicon (M1/M2/M3) or Intel Mac with Metal support
- CMake 3.14+
- LibTorch 2.0+ (with MPS support)
- Xcode Command Line Tools

## Building

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
make -j$(sysctl -n hw.ncpu)
```

## Usage

```bash
# Run training
./alphazero
```

### Training Parameters (main.cpp)

| Parameter | Default | Description |
|-----------|---------|-------------|
| ITERATIONS | 30 | Number of training iterations |
| GAMES_PER_ITER | 32 | Games played per iteration |
| MAX_SIMS | 200 | Maximum MCTS simulations per move |
| TRAIN_BATCHES | 30 | Training batches per iteration |

## Project Structure

```
.
├── CMakeLists.txt       # Build configuration
├── src/
│   ├── main.cpp         # Entry point
│   ├── Chess.h/cpp      # Chess game logic
│   ├── Network.h        # Neural network definition
│   ├── MCTS.h/mm        # Monte Carlo Tree Search (Objective-C++)
│   ├── Trainer.h/mm     # Training loop (Objective-C++)
│   └── AlignedBuffer.h  # Memory alignment utilities
└── python/
    ├── main.py          # Python training reference
    ├── mcts.py          # Python MCTS reference
    └── model.py         # Python model reference
```

## Key Optimizations

### 1. Arena Allocator
- `mmap`-based fixed memory pool (3M nodes, ~1.1GB)
- Eliminates `std::vector` reallocation overhead
- Placement new for proper `std::atomic` initialization

### 2. Lock-Free MCTS
- Virtual loss for concurrent tree exploration
- Atomic compare-exchange for node expansion
- `alignas(64)` for cache line optimization

### 3. Hybrid Inference Server
- 500μs timeout for partial batch processing
- Persistent GPU buffer to avoid allocations
- Async producer-consumer architecture

### 4. MPS Integration
- Native Metal Performance Shaders backend
- Periodic synchronization to prevent memory accumulation
- Objective-C++ for Metal API interop

## Known Issues

- Training may crash with Bus Error/Segfault under high memory pressure
- Recommended to run with 16GB+ RAM for stability
- MPS backend has known threading issues with LibTorch

## Performance

| Metric | Value |
|--------|-------|
| Throughput | ~96 moves/sec (14 workers) |
| Arena Capacity | 3M nodes (~1.1GB) |
| Batch Size | 64 |
| Inference Timeout | 500μs |

## License

MIT License

## Acknowledgments

- DeepMind's AlphaZero paper
- PyTorch/LibTorch team for MPS support
- Apple for Metal Performance Shaders
