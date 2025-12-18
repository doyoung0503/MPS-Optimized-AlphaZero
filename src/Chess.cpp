#include "Chess.h"
#include <cstring>
#include <cmath>
#include <algorithm>

Chess::Chess() {
    reset();
}

void Chess::reset() {
    board.fill(0);
    // Setup pieces
    // White: 1..6, Black: -1..-6
    // R N B Q K B N R
    int setup[] = {4, 2, 3, 5, 6, 3, 2, 4};
    for(int i=0; i<8; ++i) {
        board[i] = setup[i]; // White pieces at rank 0 ? No, rank 0 is normally black in array 0..63 usually maps to a8..h1 or a1..h8.
        // Let's use 0=a1, 63=h8 standard (little endian rank-file).
        // Rank 0 (a1..h1): White Pieces
        board[i] = setup[i];
        board[i+8] = 1; // White Pawns
        
        // Rank 7 (a8..h8): Black Pieces
        board[i+56] = -setup[i];
        board[i+48] = -1; // Black Pawns
    }
    
    turn = WHITE;
    castling.fill(true); // WK, WQ, BK, BQ
    ep_square = -1;
    halfmoves = 0;
    fullmoves = 1;
}

std::string Chess::move_to_uci(const Move& m) {
    std::string s = "";
    s += (char)('a' + (m.from % 8));
    s += (char)('1' + (m.from / 8));
    s += (char)('a' + (m.to % 8));
    s += (char)('1' + (m.to / 8));
    if (m.promotion) {
        if(m.promotion==2) s+='n';
        if(m.promotion==3) s+='b';
        if(m.promotion==4) s+='r';
        if(m.promotion==5) s+='q';
    }
    return s;
}

void Chess::print() {
    char symbols[] = ".PNBRQKpnbrqk"; // Index 0 is dot, 1..6 +ve, 7..12 -ve?? No.
    // 0 -> ., 1 -> P, ... 6 -> K. -1 -> p ... -6 -> k.
    // Map: piece -> index. 
    // abs(p) -> 1..6. if p>0 idx=p. if p<0 idx=p+6 ?? No. 
    // Let's print simply.
    
    std::cout << "  a b c d e f g h" << std::endl;
    for(int r=7; r>=0; --r) {
        std::cout << r+1 << " ";
        for(int c=0; c<8; ++c) {
            int p = board[r*8 + c];
            char c_out = '.';
            if(p > 0) c_out = " PNBRQK"[p];
            else if(p < 0) c_out = " pnbrqk"[-p]; // -(-1)=1
            // e.g. -1 (p) -> 'p'. " pnbrqk" has space at 0.
            std::cout << c_out << " ";
        }
        std::cout << r+1 << std::endl;
    }
    std::cout << "  a b c d e f g h" << std::endl;
}

bool Chess::is_square_attacked(int sq, int by_color) {
    // Check if 'sq' is attacked by any piece of 'by_color'
    // Simple reverse check: pretend a piece exists at sq and see if it hits a piece of checking type
    
    // Pawn attacks
    int forward = (by_color == WHITE) ? 1 : -1; // Direction checker comes from
    // Actually simpler: check diagonals for Pawns of 'by_color'
    // If by_color is WHITE, they are at lower ranks attacking up.
    // sq is target. P at (sq - dir +/- 1)
    
    int r = sq/8;
    int c = sq%8;
    
    // Check Pawns
    // White pawns at (r-1, c+/-1) attack (r, c)
    // Black pawns at (r+1, c+/-1) attack (r, c)
    int p_r = r - ((by_color==WHITE)?1:-1); // Rank where pawn would be
    if(p_r >=0 && p_r <8) {
        if(c-1 >=0) { if(board[p_r*8 + c-1] == 1 * by_color) return true; }
        if(c+1 <8)  { if(board[p_r*8 + c+1] == 1 * by_color) return true; }
    }
    
    // Check Knights
    int kn_offsets[] = {-17, -15, -10, -6, 6, 10, 15, 17};
    for(int off : kn_offsets) {
        int t = sq + off;
        if(t>=0 && t<64) {
            // Check file wraparound
            int tr = t/8, tc = t%8;
            if (abs(tr - r) > 2) continue; // Wrapped too far
            if (abs(tc - c) > 2) continue;
            if (board[t] == 2 * by_color) return true;
        }
    }
    
    // Check Sliding (Bishop/Queen, Rook/Queen)
    // Diagonals
    int diag_dirs[] = {-9, -7, 7, 9};
    for(int d : diag_dirs) {
        int t = sq;
        while(true) {
            int prev_c = t%8;
            t += d;
            if(t<0 || t>=64) break;
            if(abs((t%8) - prev_c) != 1) break; // Wrapped
            
            int p = board[t];
            if(p != 0) {
                if(p == 3 * by_color || p == 5 * by_color) return true;
                break; // Blocked
            }
        }
    }
    
    // Orthogonals
    int ortho_dirs[] = {-8, -1, 1, 8};
    for(int d : ortho_dirs) {
        int t = sq;
        while(true) {
            int prev_c = t%8;
            t += d;
            if(t<0 || t>=64) break;
            if(d == 1 || d == -1) { // Horizontal wrapping check
                if(abs((t%8) - prev_c) != 1) break; 
            }
            
            int p = board[t];
            if(p != 0) {
                if(p == 4 * by_color || p == 5 * by_color) return true;
                break;
            }
        }
    }
    
    // Check King
    int k_offsets[] = {-9, -8, -7, -1, 1, 7, 8, 9};
    for(int off : k_offsets) {
        int t = sq + off;
        if(t>=0 && t<64) {
            int tr = t/8, tc = t%8;
            if(abs(tr - r) <= 1 && abs(tc - c) <= 1) { // proper adj
                if(board[t] == 6 * by_color) return true;
            }
        }
    }
    
    return false;
}

bool Chess::is_in_check(int color) {
    // Find King
    int k_sq = -1;
    for(int i=0; i<64; ++i) {
        if(board[i] == 6 * color) {
            k_sq = i;
            break;
        }
    }
    if(k_sq == -1) return true; // King missing? treated as check/loss
    return is_square_attacked(k_sq, -color);
}

std::vector<Move> Chess::generate_pseudo_legal_moves() {
    std::vector<Move> moves;
    moves.reserve(50);
    
    for(int sq=0; sq<64; ++sq) {
        int p = board[sq];
        if(p == 0) continue;
        if((turn == WHITE && p < 0) || (turn == BLACK && p > 0)) continue;
        
        int type = abs(p);
        int r = sq/8;
        int c = sq%8;
        
        // Pawn
        if(type == 1) {
            int dir = (turn == WHITE) ? 1 : -1;
            int start_rank = (turn == WHITE) ? 1 : 6;
            int prom_rank = (turn == WHITE) ? 7 : 0;
            
            // Forward 1
            int f1 = sq + dir * 8;
            if(f1 >= 0 && f1 < 64 && board[f1] == 0) {
                // Promotion?
                if(r + dir == prom_rank) {
                    for(int prom : {2, 3, 4, 5}) moves.push_back({sq, f1, prom});
                } else {
                    moves.push_back({sq, f1, 0});
                    // Forward 2
                    if(r == start_rank) {
                        int f2 = sq + dir * 16;
                        if(board[f2] == 0) moves.push_back({sq, f2, 0});
                    }
                }
            }
            
            // Captures
            for(int dc : {-1, 1}) {
                if(c + dc >= 0 && c + dc < 8) {
                    int target = sq + dir * 8 + dc;
                    int tp = board[target];
                    // Regular capture
                    if((turn == WHITE && tp < 0) || (turn == BLACK && tp > 0)) {
                         if(r + dir == prom_rank) {
                            for(int prom : {2, 3, 4, 5}) moves.push_back({sq, target, prom});
                        } else {
                            moves.push_back({sq, target, 0});
                        }
                    }
                    // En Passant
                    if(target == ep_square) {
                        moves.push_back({sq, target, 0}); // Capture logic handled in make_move
                    }
                }
            }
        }
        // Knight
        else if(type == 2) {
            int offsets[] = {-17, -15, -10, -6, 6, 10, 15, 17};
            for(int off : offsets) {
                int t = sq + off;
                if(t>=0 && t<64) {
                    int tr = t/8, tc = t%8;
                    if(abs(tr-r) <= 2 && abs(tc-c) <= 2) { // check wrap
                        int tp = board[t];
                        if(tp == 0 || (turn==WHITE && tp<0) || (turn==BLACK && tp>0)) {
                            moves.push_back({sq, t, 0});
                        }
                    }
                }
            }
        }
        // King
        else if(type == 6) {
            int offsets[] = {-9, -8, -7, -1, 1, 7, 8, 9};
            for(int off : offsets) {
                int t = sq + off;
                if(t>=0 && t<64) {
                    int tr = t/8, tc = t%8;
                    if(abs(tr-r)<=1 && abs(tc-c)<=1) {
                         int tp = board[t];
                        if(tp == 0 || (turn==WHITE && tp<0) || (turn==BLACK && tp>0)) {
                            moves.push_back({sq, t, 0});
                        }
                    }
                }
            }
            // Castling
            if(!is_in_check(turn)) {
                if(turn == WHITE) {
                    // WK (K-side)
                    if(castling[0] && board[5]==0 && board[6]==0) {
                        if(!is_square_attacked(5, BLACK) && !is_square_attacked(6, BLACK))
                            moves.push_back({4, 6, 0});
                    }
                    // WQ (Q-side)
                    if(castling[1] && board[1]==0 && board[2]==0 && board[3]==0) {
                        if(!is_square_attacked(3, BLACK) /* d1 safe? not required but usually checked path */ && !is_square_attacked(2, BLACK))
                             moves.push_back({4, 2, 0}); // c1
                    }
                } else {
                    // BK
                    if(castling[2] && board[61]==0 && board[62]==0) {
                        if(!is_square_attacked(61, WHITE) && !is_square_attacked(62, WHITE))
                             moves.push_back({60, 62, 0});
                    }
                    // BQ
                    if(castling[3] && board[57]==0 && board[58]==0 && board[59]==0) {
                        if(!is_square_attacked(59, WHITE) && !is_square_attacked(58, WHITE))
                             moves.push_back({60, 58, 0});
                    }
                }
            }
        }
        // Sliding (Bishop, Rook, Queen)
        else {
             // 3=B, 4=R, 5=Q
             bool diag = (type==3 || type==5);
             bool ortho = (type==4 || type==5);
             std::vector<int> dirs;
             if(diag) for(int d : {-9, -7, 7, 9}) dirs.push_back(d);
             if(ortho) for(int d : {-8, -1, 1, 8}) dirs.push_back(d);
             
             for(int d : dirs) {
                int t = sq;
                while(true) {
                    int prev_c = t%8;
                    t += d;
                    if(t<0 || t>=64) break;
                    // Horizontal wrap check
                    if(d == 1 || d == -1) { if(abs((t%8) - prev_c) != 1) break; }
                    // Diagonal wrap check
                    if(d == -9 || d == -7 || d == 7 || d == 9) {
                         if(abs((t%8) - prev_c) != 1) break; 
                    }
                    
                    int tp = board[t];
                    if(tp == 0) {
                        moves.push_back({sq, t, 0});
                    } else {
                        if((turn==WHITE && tp<0) || (turn==BLACK && tp>0)) {
                            moves.push_back({sq, t, 0});
                        }
                        break; // Blocked
                    }
                }
             }
        }
    }
    return moves;
}

std::vector<Move> Chess::get_legal_moves() {
    std::vector<Move> pseudo = generate_pseudo_legal_moves();
    std::vector<Move> legal;
    legal.reserve(pseudo.size());
    
    for(const auto& m : pseudo) {
        // Try move
        Chess copy = *this;
        copy.push(m);
        // Check if our king is attacked (pseudo legal -> move -> check legal)
        // Wait, push() switches turn. 
        // We moved. Now it's opponent turn.
        // We verify if *opponent* can attack *our* king? No.
        // We verify if *we* left our king in check.
        // After push, turn is opponent. so we check if opponent attacks 'turn-1' king.
        
        int us = copy.turn * -1; // The player who just moved
        if(!copy.is_in_check(us)) {
            legal.push_back(m);
        }
    }
    return legal;
}

void Chess::push(const Move& m) {
    int p = board[m.from];
    int captured = board[m.to];
    
    // Move piece
    board[m.to] = p;
    board[m.from] = 0;
    
    // Promotion
    if(m.promotion) {
        board[m.to] = (turn == WHITE) ? m.promotion : -m.promotion;
    }
    
    // En Passant Capture
    if(abs(p) == 1 && m.to == ep_square) {
        // Remove pawn
        int cap_sq = m.to - ((turn == WHITE) ? 8 : -8);
        board[cap_sq] = 0;
    }
    
    // Castling Move (Move Rook)
    if(abs(p) == 6 && abs(m.from - m.to) == 2) {
        // WK
        if(m.to == 6) { board[5]=board[7]; board[7]=0; }
        // WQ
        else if(m.to == 2) { board[3]=board[0]; board[0]=0; }
        // BK
        else if(m.to == 62) { board[61]=board[63]; board[63]=0; }
        // BQ
        else if(m.to == 58) { board[59]=board[56]; board[56]=0; }
    }
    
    // Update Castling Rights
    // If King moves, lose rights
    if(p == 6) { castling[0]=false; castling[1]=false; }
    if(p == -6) { castling[2]=false; castling[3]=false; }
    // If Rook moves
    if(m.from == 0) castling[1]=false;
    if(m.from == 7) castling[0]=false;
    if(m.from == 56) castling[3]=false;
    if(m.from == 63) castling[2]=false;
    // If Rook captured (rare but important)
    if(m.to == 0) castling[1]=false;
    if(m.to == 7) castling[0]=false;
    if(m.to == 56) castling[3]=false;
    if(m.to == 63) castling[2]=false;
    
    // Update EP Square
    ep_square = -1;
    if(abs(p) == 1 && abs(m.from - m.to) == 16) {
        ep_square = (m.from + m.to) / 2;
    }
    
    // Halfmove Clock
    if(abs(p) == 1 || captured != 0) halfmoves = 0;
    else halfmoves++;
    
    if(turn == BLACK) fullmoves++;
    turn = -turn;
}

bool Chess::is_game_over() {
    return get_winner() != 2;
}

int Chess::get_winner() {
    // 1=White, -1=Black, 0=Draw, 2=Not Over
    auto moves = get_legal_moves();
    if(moves.empty()) {
        if(is_in_check(turn)) return (turn == WHITE) ? -1 : 1; // Checkmate
        return 0; // Stalemate
    }
    
    // Draw conditions
    if(halfmoves >= 100) return 0; // 50-move rule
    // Insufficient material (basic check)
    int pieces = 0;
    for(int i : board) if(i != 0) pieces++;
    if(pieces == 2) return 0; // K vs K
    
    return 2;
}

// Output array return version (wraps the optimized one)
std::array<float, 896> Chess::encode() {
    std::array<float, 896> arr;
    encode(arr.data());
    return arr;
}

void Chess::encode(float* planes) {
    // 14 planes x 8 x 8
    // Zero out
    std::fill(planes, planes + 896, 0.0f);
    
    for(int i=0; i<64; ++i) {
        int p = board[i];
        if(p != 0) {
            int plane = -1;
            int type = abs(p) - 1; // 0..5
            if(p > 0) plane = type;
            else plane = type + 6;
            
            planes[plane*64 + i] = 1.0f;
        }
    }
    
    // Castling (Plane 12)
    if(castling[0]) planes[12*64 + 63] = 1.0f;
    if(castling[1]) planes[12*64 + 56] = 1.0f;
    if(castling[2]) planes[12*64 + 7] = 1.0f;
    if(castling[3]) planes[12*64 + 0] = 1.0f;
    
    // EP (Plane 13)
    if(ep_square != -1) {
        planes[13*64 + ep_square] = 1.0f;
    }
}


Chess Chess::get_canonical() {
    if(turn == WHITE) return *this;
    Chess c = *this;
    c.mirror();
    return c;
}

void Chess::mirror() {
    // 1. Flip Board Vertical (Rank 0<->7, 1<->6 ...)
    std::array<int, 64> new_board;
    new_board.fill(0);
    
    for(int i=0; i<64; ++i) {
        int r = i / 8;
        int c = i % 8;
        int new_r = 7 - r;
        int new_i = new_r * 8 + c;
        
        int p = board[i];
        if(p != 0) {
            new_board[new_i] = -p; // Swap Color
        }
    }
    board = new_board;
    
    // 2. Flip Castling Rights
    // castling: [WK, WQ, BK, BQ]
    // Swap White <-> Black rights
    bool tmp_k = castling[0];
    bool tmp_q = castling[1];
    castling[0] = castling[2];
    castling[1] = castling[3];
    castling[2] = tmp_k;
    castling[3] = tmp_q;
    
    // 3. Flip EP Square
    if(ep_square != -1) {
        int r = ep_square / 8;
        int c = ep_square % 8;
        ep_square = (7 - r) * 8 + c;
    }
    
    // 4. Swap Turn
    turn = -turn;
}
