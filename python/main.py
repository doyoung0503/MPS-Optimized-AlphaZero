"""
AlphaZero Training - Main Entry Point
"""
import torch
from network import AlphaZeroNet
from trainer import Trainer

def main():
    # Device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Model
    model = AlphaZeroNet(
        in_channels=14,
        board_size=8,
        action_size=4352,
        num_res_blocks=6,
        filters=64
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer
    trainer = Trainer(model, device=device, lr=0.001)
    
    # Training configuration
    iterations = 20
    games_per_iter = 64
    simulations = 50
    train_batches = 20
    
    # Train
    trainer.train(
        iterations=iterations,
        games_per_iter=games_per_iter,
       max_sims=simulations,
        train_batches=train_batches
    )
    
    # Save model
    torch.save(model.state_dict(), 'alphazero_python.pth')
    print("\nModel saved to alphazero_python.pth")

if __name__ == '__main__':
    main()
