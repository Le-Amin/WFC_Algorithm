#include <iostream>
#include <string>
#include <chrono>
#include <stdexcept>
#include <cstdlib>
#include "grid.hpp"
#include "wfc.hpp"

// Taille des tuiles (paramètre compile-time)
static constexpr int TILE_SIZE = 2;

// Affiche l'usage
static void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <sample> <out_rows> <out_cols> [mode] [seed]\n"
              << "  sample   : fichier texte de la grille échantillon\n"
              << "  out_rows : hauteur de la grille de sortie\n"
              << "  out_cols : largeur de la grille de sortie\n"
              << "  mode     : serial (défaut) | omp\n"
              << "  seed     : graine aléatoire (défaut: 42)\n";
}

int main(int argc, char** argv) {
    if (argc < 4) { usage(argv[0]); return 1; }

    std::string sample_path = argv[1];
    int out_rows = std::atoi(argv[2]);
    int out_cols = std::atoi(argv[3]);
    std::string mode = (argc >= 5) ? argv[4] : "serial";
    unsigned seed    = (argc >= 6) ? static_cast<unsigned>(std::atoi(argv[5])) : 42u;

    if (out_rows <= 0 || out_cols <= 0) {
        std::cerr << "Erreur : dimensions invalides\n"; return 1;
    }

    try {
        // Chargement de la grille échantillon
        auto sample = Grid<int>::load(sample_path);
        std::cout << "Échantillon chargé : "
                  << sample.rows << "×" << sample.cols << "\n";

        // Construction du WFC
        WFC<TILE_SIZE, int> wfc(sample);
        std::cout << "Tuiles valides : " << wfc.num_tiles << "\n";

        Grid<int> output;
        bool ok = false;

        auto t0 = std::chrono::high_resolution_clock::now();

        if (mode == "omp") {
            std::cout << "Mode : OpenMP tasks\n";
            ok = wfc.solve_omp(output, out_rows, out_cols, seed);
        } else {
            std::cout << "Mode : série\n";
            ok = wfc.solve_serial(output, out_rows, out_cols, seed);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (!ok) {
            std::cerr << "Échec de la résolution (contradiction).\n";
            return 2;
        }

        std::cout << "Résolution OK en " << elapsed_ms << " ms\n";
        output.print();

        // Sauvegarde
        std::string out_path = "results/output_" + mode + ".txt";
        output.save(out_path);
        std::cout << "Résultat sauvegardé dans " << out_path << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << "\n";
        return 1;
    }

    return 0;
}
