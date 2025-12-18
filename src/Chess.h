#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <array>

// Pieces: 0=Empty, 1=Pawn, 2=Knight, 3=Bishop, 4=Rook, 5=Queen, 6=King
// Colors: White=0, Black=1. stored as 10 + piece for White, 20 + piece for Black.
// Actually simpler: Positive for White, Negative for Black.
// 1..6 = White, -1..-6 = Black. 0 = Empty.

enum Player { WHITE = 1, BLACK = -1 };

struct Move {
    int from; // 0-63
    int to;   // 0-63
    int promotion; // 0, or piece type (2,3,4,5)
    
    // Equality
    bool operator==(const Move& other) const {
        return from == other.from && to == other.to && promotion == other.promotion;
    }
};

class Chess {
public:
    // Board representation: 64 integers
    std::array<int, 64> board;
    int turn; // 1 or -1
    
    // Castling rights: WK, WQ, BK, BQ
    std::array<bool, 4> castling;
    
    // En passant square: -1 or 0-63
    int ep_square;
    
    // Halfmove clock for 50-move rule
    int halfmoves;
    
    // Fullmove number
    int fullmoves;
    
    // Explicit padding for 8-byte alignment on ARM64
    // Current size: 64*4 + 4 + 4 + 4 + 4 + 4 = 276 bytes
    // Add 4 bytes to reach 280 (divisible by 8)
    char _padding[4] = {0};

    Chess();
    
    void reset();
    
    // Core Logic
    std::vector<Move> get_legal_moves();
    void push(const Move& move);
    
    bool is_game_over();
    int get_winner(); // 1=White, -1=Black, 0=Draw, 2=Not Over
    
    // Helpers
    static std::string move_to_uci(const Move& m);
    static Move uci_to_move(const std::string& uci);
    void print();
    
    // Neural Network Helpers
    std::array<float, 896> encode(); // Keep old for compatibility or tiny tests
    void encode(float* out);          // Optimized version for batching
    Chess get_canonical();
    void mirror(); // Flip board vertical + swap colors
    
private:
    std::vector<Move> generate_pseudo_legal_moves();
    bool is_in_check(int color);
    bool is_square_attacked(int sq, int by_color);
};
