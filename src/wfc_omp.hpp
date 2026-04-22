#pragma once
#include <omp.h>
// Inclus via wfc.hpp — ne pas inclure directement

// ============================================================
// Version parallèle OpenMP avec API task explicite
//
// Stratégie de parallélisation :
//  - La boucle principale (observation + collapse) reste série
//    car elle dépend de l'état courant de la wave.
//  - La PROPAGATION utilise des tâches OpenMP :
//    chaque cellule voisine à propager est traitée comme une tâche.
//  - Le calcul de l'entropie (find min) est parallélisé via
//    omp parallel for reduction.
// ============================================================

template<int N, typename T>
bool WFC<N, T>::solve_omp(Grid<T>& output, int rows, int cols,
                           unsigned seed) const
{
    if (num_tiles == 0) return false;

    output = Grid<T>(rows, cols, T{});
    Wave wave = init_wave(rows * cols);
    std::mt19937 rng(seed);

    int total_cells = rows * cols;
    bool contradiction = false;

    while (!contradiction) {
        // --- Recherche parallèle de la cellule de plus faible entropie ---
        int min_entropy = std::numeric_limits<int>::max();
        int chosen_cell = -1;

        #pragma omp parallel for schedule(static) \
                shared(wave, min_entropy, chosen_cell, contradiction)
        for (int cell = 0; cell < total_cells; ++cell) {
            if (contradiction) continue;
            int e = entropy(cell, wave);
            if (e == 0) {
                #pragma omp critical
                contradiction = true;
                continue;
            }
            if (e == 1) continue; // déjà collapsée

            #pragma omp critical
            {
                if (e < min_entropy) {
                    min_entropy = e;
                    chosen_cell = cell;
                }
            }
        }

        if (contradiction) return false;
        if (chosen_cell < 0) break; // tout collapsé

        // --- Collapse (série — dépend du RNG et de l'état global) ---
        int chosen_tile = choose_tile(chosen_cell, wave, rng);
        if (chosen_tile < 0) return false;
        collapse(chosen_cell, chosen_tile, wave);

        // --- Propagation par tâches OpenMP ---
        // On utilise une queue protégée par un verrou.
        // Chaque cellule à propager est soumise comme une tâche.

        omp_lock_t queue_lock;
        omp_init_lock(&queue_lock);
        bool prop_ok = true;

        std::queue<int> pending;
        pending.push(chosen_cell);

        #pragma omp parallel shared(pending, wave, prop_ok, queue_lock)
        {
            #pragma omp single nowait
            {
                while (!pending.empty() && prop_ok) {
                    omp_set_lock(&queue_lock);
                    if (pending.empty()) { omp_unset_lock(&queue_lock); break; }
                    int cell = pending.front(); pending.pop();
                    omp_unset_lock(&queue_lock);

                    int cr = cell / cols, cc = cell % cols;

                    for (int oy = -(N-1); oy <= N-1; ++oy) {
                        for (int ox = -(N-1); ox <= N-1; ++ox) {
                            if (ox == 0 && oy == 0) continue;
                            int nr = cr + oy, nc = cc + ox;
                            if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;

                            int ncell = nr * cols + nc;
                            int rev_ox = -ox, rev_oy = -oy;

                            // Chaque voisin → une tâche OpenMP
                            #pragma omp task firstprivate(cell, ncell, rev_ox, rev_oy) \
                                             shared(wave, pending, prop_ok, queue_lock)
                            {
                                bool changed = false;
                                bool local_ok = true;

                                // Section critique pour lire/écrire la wave du voisin
                                #pragma omp critical(wave_update)
                                {
                                    for (int t2 = 0; t2 < num_tiles; ++t2) {
                                        if (!wave[ncell][t2]) continue;

                                        bool supported = false;
                                        const auto& allowed =
                                            compatible[t2][rev_oy + OFF][rev_ox + OFF];
                                        for (int t1 : allowed) {
                                            if (wave[cell][t1]) {
                                                supported = true; break;
                                            }
                                        }
                                        if (!supported) {
                                            wave[ncell][t2] = false;
                                            changed = true;
                                        }
                                    }
                                    if (changed && entropy(ncell, wave) == 0)
                                        local_ok = false;
                                }

                                if (!local_ok) {
                                    #pragma omp critical
                                    prop_ok = false;
                                } else if (changed) {
                                    omp_set_lock(&queue_lock);
                                    pending.push(ncell);
                                    omp_unset_lock(&queue_lock);
                                }
                            } // fin task
                        }
                    }
                    #pragma omp taskwait
                } // fin while
            } // fin single
        } // fin parallel

        omp_destroy_lock(&queue_lock);
        if (!prop_ok) return false;
    }

    return extract_output(wave, output, rows, cols);
}
