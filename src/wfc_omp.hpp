#pragma once
#include <omp.h>
#include <vector>
#include <queue>
#include <limits>

// ============================================================
// WFC::solve_omp  —  version parallèle OpenMP (API task)
//
// Architecture :
//   1. Recherche de la cellule min-entropie  → omp parallel for
//   2. Collapse                              → série (RNG global)
//   3. Propagation BFS                       → omp task par voisin
//
// Invariant de sécurité (propagation sans verrou) :
//   Pour un src donné, chaque tâche j écrit dans wave[dst_j] où
//   dst_j = src + (oy_j, ox_j).  Les offsets étant distincts, les
//   dst_j sont TOUS DISTINCTS → pas de race condition en écriture.
//   wave[src] est en lecture seule pendant les tâches → lecture
//   concurrente sûre.  La file BFS est mise à jour en série après
//   taskwait → pas de verrou nécessaire sur la file.
// ============================================================

template<int N, typename T>
bool WFC<N, T>::solve_omp(Grid<T>& output, int rows, int cols,
                           unsigned seed) const
{
    if (num_tiles == 0) return false;

    output = Grid<T>(rows, cols, T{});
    Wave wave = init_wave(rows * cols);
    std::mt19937 rng(seed);
    const int total_cells = rows * cols;

    while (true) {

        // ── 1. Recherche parallèle de la cellule de plus faible entropie ──
        int chosen_cell = -1;
        int  g_min    = std::numeric_limits<int>::max();
        bool g_contra = false;

        #pragma omp parallel shared(chosen_cell, g_min, g_contra)
        {
            int  t_min    = std::numeric_limits<int>::max();
            int  t_cell   = -1;
            bool t_contra = false;

            #pragma omp for nowait schedule(static)
            for (int c = 0; c < total_cells; ++c) {
                int e = entropy(c, wave);
                if (e == 0)                  { t_contra = true; }
                else if (e > 1 && e < t_min) { t_min = e; t_cell = c; }
            }

            #pragma omp critical(entropy_reduce)
            {
                if (t_contra)      g_contra = true;
                if (t_min < g_min) { g_min = t_min; chosen_cell = t_cell; }
            }
        }

        if (g_contra)        return false;
        if (chosen_cell < 0) break;   // tout collapsé → succès

        // ── 2. Collapse (série — dépend du RNG) ───────────────────────────
        int chosen_tile = choose_tile(chosen_cell, wave, rng);
        if (chosen_tile < 0) return false;
        collapse(chosen_cell, chosen_tile, wave);

        // ── 3. Propagation par tâches OpenMP (BFS) ────────────────────────
        // Chaque étape BFS traite un src et lance une tâche par voisin valide.
        // Les tâches écrivent dans des cellules disjointes → aucun verrou sur
        // wave[]. Les résultats (changed / contradiction) sont collectés dans
        // des tableaux indexés par j (uniques par tâche), puis la file BFS est
        // mise à jour en série après le taskwait.

        struct Job { int dst, rev_ox, rev_oy; };

        std::queue<int>   bfs;
        std::vector<bool> visited(total_cells, false);
        bfs.push(chosen_cell);
        visited[chosen_cell] = true;

        while (!bfs.empty()) {
            const int src = bfs.front(); bfs.pop();
            const int cr  = src / cols,  cc = src % cols;

            // Construire la liste des voisins valides de src
            std::vector<Job> jobs;
            jobs.reserve((2*N-1)*(2*N-1) - 1);
            for (int oy = -(N-1); oy <= N-1; ++oy)
                for (int ox = -(N-1); ox <= N-1; ++ox) {
                    if (ox == 0 && oy == 0) continue;
                    int nr = cr + oy, nc = cc + ox;
                    if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
                    jobs.push_back({nr * cols + nc, -ox, -oy});
                }

            const int njobs = static_cast<int>(jobs.size());

            // Tableaux de résultats indexés par tâche (pas de race : j unique)
            //   result[j] = 0  → pas de changement
            //   result[j] = 1  → wave[dst] a changé (à remettre en file)
            //   result[j] = -1 → contradiction (wave[dst] vide)
            std::vector<int> result(njobs, 0);

            #pragma omp parallel shared(wave, jobs, result)
            #pragma omp single nowait
            {
                for (int j = 0; j < njobs; ++j) {
                    #pragma omp task firstprivate(j) shared(wave, jobs, result)
                    {
                        const Job& jb = jobs[j];
                        bool cell_changed = false;

                        for (int t2 = 0; t2 < num_tiles; ++t2) {
                            if (!wave[jb.dst][t2]) continue;
                            bool supported = false;
                            // Lecture de wave[src] : read-only ici → sûr en concurrence
                            for (int t1 : compatible[t2][jb.rev_oy+OFF][jb.rev_ox+OFF])
                                if (wave[src][t1]) { supported = true; break; }
                            if (!supported) {
                                wave[jb.dst][t2] = false;
                                cell_changed = true;
                            }
                        }

                        if (cell_changed) {
                            int cnt = 0;
                            for (int t = 0; t < num_tiles; ++t)
                                if (wave[jb.dst][t]) ++cnt;
                            result[j] = (cnt == 0) ? -1 : 1;
                        }
                    } // fin task
                }
                #pragma omp taskwait
            } // fin single + parallel

            // Mise à jour de la file BFS (série — après taskwait)
            for (int j = 0; j < njobs; ++j) {
                if (result[j] == -1)                       return false;
                if (result[j] ==  1 && !visited[jobs[j].dst]) {
                    visited[jobs[j].dst] = true;
                    bfs.push(jobs[j].dst);
                }
            }
        } // fin BFS
    } // fin boucle principale

    return extract_output(wave, output, rows, cols);
}
