#pragma once
#include <array>
#include <algorithm>
#include "grid.hpp"

// Tuile carrée de taille N×N avec valeurs de type T
template<int N, typename T>
struct Tile {
    std::array<T, N * N> data{};
    int frequency = 1;

    T  operator()(int r, int c) const { return data[r * N + c]; }
    T& operator()(int r, int c)       { return data[r * N + c]; }

    bool operator==(const Tile& o) const { return data == o.data; }
    bool operator!=(const Tile& o) const { return data != o.data; }

    // Extrait la tuile N×N dont le coin supérieur gauche est en (row, col) dans grid
    static Tile extract(const Grid<T>& grid, int row, int col) {
        Tile t;
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c)
                t(r, c) = grid(row + r, col + c);
        return t;
    }

    // Retourne vrai si *this (position p) et other (position p + (dx,dy))
    // sont compatibles : les régions qui se chevauchent ont les mêmes valeurs.
    //
    // dx, dy ∈ [-(N-1), N-1]
    // La zone de chevauchement dans *this : r ∈ [max(0,dy), min(N, N+dy)[
    //                                       c ∈ [max(0,dx), min(N, N+dx)[
    // La position correspondante dans other : (r-dy, c-dx)
    bool compatible(const Tile& other, int dx, int dy) const {
        int r_lo = std::max(0, dy),  r_hi = std::min(N, N + dy);
        int c_lo = std::max(0, dx),  c_hi = std::min(N, N + dx);
        for (int r = r_lo; r < r_hi; ++r)
            for (int c = c_lo; c < c_hi; ++c)
                if ((*this)(r, c) != other(r - dy, c - dx))
                    return false;
        return true;
    }

    void print() const {
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                if (c) std::cout << ' ';
                std::cout << static_cast<int>((*this)(r, c));
            }
            std::cout << '\n';
        }
    }
};
