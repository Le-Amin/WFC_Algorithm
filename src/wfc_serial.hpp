#pragma once
// Inclus via wfc.hpp — ne pas inclure directement

template<int N, typename T>
bool WFC<N, T>::solve_serial(Grid<T>& output, int rows, int cols,
                              unsigned seed) const
{
    if (num_tiles == 0) return false;

    output = Grid<T>(rows, cols, T{});
    Wave wave = init_wave(rows * cols);
    std::mt19937 rng(seed);

    int total_cells = rows * cols;
    int collapsed   = 0;

    while (collapsed < total_cells) {
        // --- Étape 4 : choisir la cellule de plus faible entropie > 1 ---
        int min_entropy = std::numeric_limits<int>::max();
        int chosen_cell = -1;

        for (int cell = 0; cell < total_cells; ++cell) {
            int e = entropy(cell, wave);
            if (e == 0) return false;          // contradiction
            if (e == 1) continue;              // déjà collapsée
            if (e < min_entropy) {
                min_entropy = e;
                chosen_cell = cell;
            }
        }
        if (chosen_cell < 0) break;            // tout est collapsé

        // --- Étape 4 (suite) : choisir une tuile aléatoirement ---
        int chosen_tile = choose_tile(chosen_cell, wave, rng);
        if (chosen_tile < 0) return false;

        collapse(chosen_cell, chosen_tile, wave);
        ++collapsed;

        // --- Étape 5 : propager les contraintes ---
        if (!propagate(chosen_cell, wave, rows, cols)) return false;
    }

    return extract_output(wave, output, rows, cols);
}
