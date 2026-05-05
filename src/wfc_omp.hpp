#pragma once
#include <omp.h>
#include <vector>
#include <queue>
#include <limits>

// ============================================================
// WFC::solve_omp  —  version parallèle OpenMP (API task)
//
// Architecture : UNE SEULE région parallèle pour tout l'algo.
//   Les threads sont créés une fois et restent actifs.
//
//   Chaque itération de la boucle principale :
//     1. Entropie   → #pragma omp for   (tous les threads)
//     2. Collapse   → #pragma omp single (un thread)
//     3. BFS + tâches → #pragma omp task (un thread génère,
//                        les autres exécutent)
//
// Pourquoi une seule région ?
//   Créer/détruire une région parallèle à chaque étape BFS
//   coûte O(ms) (réveil des threads). Avec 8 voisins par cellule
//   et des milliers d'étapes, cet overhead dominait le calcul.
//
// Invariant de sécurité (tâches sans verrou) :
//   Pour un src fixé, dst_j = src+(oy_j,ox_j) sont tous DISTINCTS
//   → les tâches écrivent dans des cellules disjointes de wave[].
//   wave[src] est en lecture seule pendant les tâches → sûr.
//   Les résultats sont dans result[j] (index unique) → pas de race.
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

    // Variables partagées entre les threads
    int  g_chosen = -1;
    int  g_min    = std::numeric_limits<int>::max();
    bool g_contra = false;
    bool g_done   = false;
    bool g_result = true;

    // ── Unique région parallèle ────────────────────────────────────────
    #pragma omp parallel shared(wave, g_chosen, g_min, g_contra, g_done, g_result)
    {
        while (!g_done) {

            // ── 1. Recherche parallèle de la cellule min-entropie ─────
            int  t_min    = std::numeric_limits<int>::max();
            int  t_cell   = -1;
            bool t_contra = false;

            #pragma omp for nowait schedule(static)
            for (int c = 0; c < total_cells; ++c) {
                int e = entropy(c, wave);
                if (e == 0)                  t_contra = true;
                else if (e > 1 && e < t_min) { t_min = e; t_cell = c; }
            }

            #pragma omp critical(reduce)
            {
                if (t_contra)      g_contra = true;
                if (t_min < g_min) { g_min = t_min; g_chosen = t_cell; }
            }
            #pragma omp barrier  // tous les threads ont réduit

            // ── 2. Collapse + propagation BFS (un thread + tâches) ────
            // Le bloc single a une barrière implicite à la sortie :
            // tous les threads attendent avant l'itération suivante.
            #pragma omp single
            {
                if (g_contra || !g_result) {
                    g_result = false;
                    g_done   = true;
                } else if (g_chosen < 0) {
                    g_done = true;       // tout collapsé → succès
                } else {
                    int tile = choose_tile(g_chosen, wave, rng);
                    if (tile < 0) { g_result = false; g_done = true; }
                    else {
                        collapse(g_chosen, tile, wave);

                        // ── BFS avec tâches ───────────────────────────
                        struct Job { int dst, rev_ox, rev_oy; };

                        std::queue<int>   bfs;
                        std::vector<bool> in_queue(total_cells, false);
                        bfs.push(g_chosen);
                        in_queue[g_chosen] = true;
                        bool prop_ok = true;

                        while (!bfs.empty() && prop_ok) {
                            const int src = bfs.front(); bfs.pop();
                            in_queue[src] = false;
                            const int cr = src / cols, cc = src % cols;

                            // Construire la liste des voisins valides
                            std::vector<Job> jobs;
                            jobs.reserve((2*N-1)*(2*N-1) - 1);
                            for (int oy = -(N-1); oy <= N-1; ++oy)
                                for (int ox = -(N-1); ox <= N-1; ++ox) {
                                    if (ox == 0 && oy == 0) continue;
                                    int nr = cr+oy, nc = cc+ox;
                                    if (nr<0||nr>=rows||nc<0||nc>=cols) continue;
                                    jobs.push_back({nr*cols+nc, -ox, -oy});
                                }

                            const int njobs = static_cast<int>(jobs.size());
                            // result[j] : 0=pas de changement, 1=changé, -1=contradiction
                            std::vector<int> result(njobs, 0);

                            // Une tâche par voisin — les autres threads les exécutent
                            for (int j = 0; j < njobs; ++j) {
                                #pragma omp task firstprivate(j) \
                                                 shared(wave, jobs, result)
                                {
                                    const Job& jb = jobs[j];
                                    bool changed = false;
                                    for (int t2 = 0; t2 < num_tiles; ++t2) {
                                        if (!wave[jb.dst][t2]) continue;
                                        bool supported = false;
                                        for (int t1 : compatible[t2][jb.rev_oy+OFF][jb.rev_ox+OFF])
                                            if (wave[src][t1]) { supported = true; break; }
                                        if (!supported) {
                                            wave[jb.dst][t2] = false;
                                            changed = true;
                                        }
                                    }
                                    if (changed) {
                                        int cnt = 0;
                                        for (int t = 0; t < num_tiles; ++t)
                                            if (wave[jb.dst][t]) ++cnt;
                                        result[j] = (cnt == 0) ? -1 : 1;
                                    }
                                } // fin task
                            }
                            #pragma omp taskwait

                            // Mise à jour BFS (série, après taskwait)
                            for (int j = 0; j < njobs; ++j) {
                                if (result[j] == -1) { prop_ok = false; break; }
                                if (result[j] ==  1 && !in_queue[jobs[j].dst]) {
                                    in_queue[jobs[j].dst] = true;
                                    bfs.push(jobs[j].dst);
                                }
                            }
                        } // fin BFS

                        if (!prop_ok) { g_result = false; g_done = true; }
                    }
                }

                // Réinitialiser pour la prochaine itération
                g_min    = std::numeric_limits<int>::max();
                g_contra = false;
                g_chosen = -1;
            } // fin single  (barrière implicite → tous les threads synchronisés)

        } // fin while
    } // fin parallel

    if (!g_result) return false;
    return extract_output(wave, output, rows, cols);
}
