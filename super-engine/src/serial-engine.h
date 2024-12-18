#ifndef SERIAL_ENGINE_H
#define SERIAL_ENGINE_H

#include "thc.h"      // Include the THC library header
#include <chrono>
#include <atomic>
#include <vector>     // For std::vector
#include <cstdint>
#include <random>

class SerialEngine {
public:
    using Score = float;

    SerialEngine();
    ~SerialEngine();

    // Solve function to find the best move
    thc::Move solve(thc::ChessRules& cr, bool is_white_player);

private:
    // Recursive search function with alpha-beta pruning and iterative deepening
    Score solve_serial_engine(
        thc::ChessRules& cr,
        bool is_white_player,
        thc::Move& best_move,
        int depth,
        int max_depth,
        Score alpha_score,
        Score beta_score
    );

    static constexpr Score INF_SCORE = 1000000.0f;
    static constexpr int MAX_DEPTH = 8;
    static constexpr int TIME_LIMIT_SECONDS = 200; // Time limit in seconds

    // Static evaluation function
    Score static_eval(thc::ChessRules& cr);

    // Helper function to score moves for move ordering
    float score_move(const thc::Move& move, thc::ChessRules& cr);

    // **Add the missing function declarations here**

    // Function to evaluate mobility
    int evaluate_mobility(thc::ChessRules& cr, bool is_white, const std::vector<int>& piece_indices);

    // Function to evaluate pawn structure
    int evaluate_pawn_structure(const std::vector<int>& pawn_files, bool is_white);

    // Function to evaluate king safety
    int evaluate_king_safety(thc::ChessRules& cr, int king_index, bool is_white, bool endgame);

    // Function to detect endgame phase
    bool is_endgame(int white_material, int black_material);

    // Function to evaluate king activity in endgame
    int evaluate_king_activity(int own_king_index, int opponent_king_index, bool is_white);

    Score quiesce(thc::ChessRules &cr, Score alpha, Score beta);

    uint64_t zobrist[12][64];

    // Additional Zobrist keys:
    uint64_t zobrist_side;                   // For side to move
    uint64_t zobrist_castling[16];           // For castling rights (4 bits: White-K, White-Q, Black-K, Black-Q)
    uint64_t zobrist_en_passant[8];          // For en passant file (0-7 if any)


    struct TTEntry {
        uint64_t key;
        int depth;
        int score;
        thc::Move best_move;
        enum BoundType { BOUND_EXACT, BOUND_LOWER, BOUND_UPPER } bound;
    };

    static constexpr size_t TT_SIZE = 1 << 20; 
    std::vector<TTEntry> transposition_table { TT_SIZE };


    // Time management variables
    std::chrono::steady_clock::time_point start_time;
    std::atomic<bool> time_limit_reached;

    void init_zobrist();
    uint64_t compute_zobrist_key(const thc::ChessRules& cr);
    void init_transposition_table();
    TTEntry* probe_tt(uint64_t key);
    void store_tt(uint64_t key, int depth, int score, TTEntry::BoundType bound, const thc::Move& best_move);
};

#endif // SERIAL_ENGINE_H
