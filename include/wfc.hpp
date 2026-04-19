#pragma once
#include <vector>
#include <random>
#include <stdexcept>
#include <queue>
#include <functional>
#include <limits>
#include <cmath>
#include <algorithm>
#include "grid.hpp"
#include "tile.hpp"

// ============================================================
// WFC<N, T> : Wave Function Collapse (modèle overlapping)
//
// N : taille des tuiles (paramètre template)
// T : type des valeurs de la grille (uint8_t, int, …)
// ============================================================
template<int N, typename T>
class WFC {
public:
    // ----------------------------------------------------------
    // Tuiles extraites de l'échantillon
    // ----------------------------------------------------------
    std::vector<Tile<N, T>> tiles;   // tuiles uniques
    int num_tiles = 0;

    // ----------------------------------------------------------
    // Règles de compatibilité
    // compatible[t1][oy][ox] = liste des indices de tuiles
    // compatibles avec t1 pour l'offset (ox, oy).
    // ox, oy ∈ [-(N-1), N-1] → index = oy+(N-1), ox+(N-1)
    // ----------------------------------------------------------
    static constexpr int OFF = N - 1;          // décalage d'indexation
    static constexpr int DIM = 2 * N - 1;      // taille de la dimension offset

    // compatible[t1_idx][oy_idx][ox_idx] = vecteur d'indices de tuiles compatibles
    std::vector<std::vector<std::vector<std::vector<int>>>> compatible;

    // ----------------------------------------------------------
    // Constructeur : analyse l'échantillon
    // ----------------------------------------------------------
    explicit WFC(const Grid<T>& sample) {
        extract_tiles(sample);
        compute_compatibility();
    }

    // ----------------------------------------------------------
    // Résolution série
    // Remplit output (rows×cols) et retourne true si succès.
    // ----------------------------------------------------------
    bool solve_serial(Grid<T>& output, int rows, int cols,
                      unsigned seed = 42) const;

    // ----------------------------------------------------------
    // Résolution parallèle (OpenMP tasks)
    // ----------------------------------------------------------
    bool solve_omp(Grid<T>& output, int rows, int cols,
                   unsigned seed = 42) const;

private:
    // ------ Construction -----------------------------------------------

    void extract_tiles(const Grid<T>& sample) {
        for (int r = 0; r <= sample.rows - N; ++r) {
            for (int c = 0; c <= sample.cols - N; ++c) {
                Tile<N, T> t = Tile<N, T>::extract(sample, r, c);
                auto it = std::find(tiles.begin(), tiles.end(), t);
                if (it == tiles.end()) {
                    tiles.push_back(t);
                } else {
                    it->frequency++;
                }
            }
        }
        num_tiles = static_cast<int>(tiles.size());
    }

    void compute_compatibility() {
        // Allouer : [num_tiles][DIM][DIM] → vector d'indices
        compatible.assign(num_tiles,
            std::vector<std::vector<std::vector<int>>>(DIM,
                std::vector<std::vector<int>>(DIM)));

        for (int t1 = 0; t1 < num_tiles; ++t1) {
            for (int oy = -(N-1); oy <= N-1; ++oy) {
                for (int ox = -(N-1); ox <= N-1; ++ox) {
                    if (ox == 0 && oy == 0) continue;
                    for (int t2 = 0; t2 < num_tiles; ++t2) {
                        if (tiles[t1].compatible(tiles[t2], ox, oy))
                            compatible[t1][oy + OFF][ox + OFF].push_back(t2);
                    }
                }
            }
        }
    }

    // ------ État de résolution -----------------------------------------

    // wave[cell] = vecteur de booléens (taille num_tiles)
    //   wave[cell][t] = true  → la tuile t est encore possible pour cette cellule
    using Wave = std::vector<std::vector<bool>>;

    Wave init_wave(int cells) const {
        return Wave(cells, std::vector<bool>(num_tiles, true));
    }

    // Entropie d'une cellule = nombre de tuiles encore possibles
    // Retourne 0 si contradiction, -1 si déjà collapsée (1 tuile)
    int entropy(int cell, const Wave& wave) const {
        int count = 0;
        for (int t = 0; t < num_tiles; ++t)
            if (wave[cell][t]) ++count;
        return count;
    }

    // Choisit aléatoirement une tuile parmi les possibles (pondérée par fréquence)
    int choose_tile(int cell, const Wave& wave, std::mt19937& rng) const {
        int total_freq = 0;
        for (int t = 0; t < num_tiles; ++t)
            if (wave[cell][t]) total_freq += tiles[t].frequency;

        std::uniform_int_distribution<int> dist(0, total_freq - 1);
        int r = dist(rng);
        for (int t = 0; t < num_tiles; ++t) {
            if (wave[cell][t]) {
                r -= tiles[t].frequency;
                if (r < 0) return t;
            }
        }
        return -1; // ne devrait pas arriver
    }

    // Propagation par contraintes (AC-3 simplifié)
    // Retourne false en cas de contradiction
    bool propagate(int start_cell, Wave& wave, int rows, int cols) const {
        std::queue<int> queue;
        queue.push(start_cell);

        while (!queue.empty()) {
            int cell = queue.front(); queue.pop();
            int cr = cell / cols, cc = cell % cols;

            for (int oy = -(N-1); oy <= N-1; ++oy) {
                for (int ox = -(N-1); ox <= N-1; ++ox) {
                    if (ox == 0 && oy == 0) continue;
                    int nr = cr + oy, nc = cc + ox;
                    if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
                    int ncell = nr * cols + nc;

                    bool changed = false;
                    for (int t2 = 0; t2 < num_tiles; ++t2) {
                        if (!wave[ncell][t2]) continue;

                        // t2 est-il compatible avec au moins une tuile possible de cell ?
                        bool supported = false;
                        // Offset de cell par rapport à ncell : (-ox, -oy)
                        int rev_ox = -ox, rev_oy = -oy;
                        const auto& allowed = compatible[t2][rev_oy + OFF][rev_ox + OFF];
                        for (int t1 : allowed) {
                            if (wave[cell][t1]) { supported = true; break; }
                        }
                        if (!supported) {
                            wave[ncell][t2] = false;
                            changed = true;
                        }
                    }
                    if (changed) {
                        if (entropy(ncell, wave) == 0) return false; // contradiction
                        queue.push(ncell);
                    }
                }
            }
        }
        return true;
    }

    // Collapse la cellule cell à la tuile chosen_tile
    void collapse(int cell, int chosen_tile, Wave& wave) const {
        for (int t = 0; t < num_tiles; ++t)
            wave[cell][t] = (t == chosen_tile);
    }

    // Extrait la valeur [0][0] de la tuile assignée à chaque cellule
    bool extract_output(const Wave& wave, Grid<T>& output, int rows, int cols) const {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                int cell = r * cols + c;
                int assigned = -1;
                for (int t = 0; t < num_tiles; ++t) {
                    if (wave[cell][t]) { assigned = t; break; }
                }
                if (assigned < 0) return false;
                output(r, c) = tiles[assigned](0, 0);
            }
        }
        return true;
    }
};

// ============================================================
// Inclusion des implémentations (séparées pour la lisibilité)
// ============================================================
#include "../src/wfc_serial.hpp"
#ifdef _OPENMP
#  include "../src/wfc_omp.hpp"
#else
// Repli : solve_omp délègue à solve_serial si OpenMP absent
template<int N, typename T>
bool WFC<N,T>::solve_omp(Grid<T>& output, int rows, int cols, unsigned seed) const {
    return solve_serial(output, rows, cols, seed);
}
#endif
