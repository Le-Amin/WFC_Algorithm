#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <iomanip>

// Grille 2D générique (lignes × colonnes)
template<typename T>
class Grid {
public:
    int rows, cols;
    std::vector<T> data;

    Grid() : rows(0), cols(0) {}

    Grid(int rows, int cols, T init = T{})
        : rows(rows), cols(cols), data(rows * cols, init) {}

    T& operator()(int r, int c)             { return data[r * cols + c]; }
    T  operator()(int r, int c) const       { return data[r * cols + c]; }

    bool in_bounds(int r, int c) const {
        return r >= 0 && r < rows && c >= 0 && c < cols;
    }

    // Chargement depuis fichier texte (valeurs séparées par espaces, une ligne par rangée)
    static Grid<T> load(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("Impossible d'ouvrir : " + path);

        std::vector<std::vector<T>> tmp;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            std::vector<T> row;
            int v;
            while (iss >> v) row.push_back(static_cast<T>(v));
            if (!row.empty()) tmp.push_back(row);
        }

        if (tmp.empty()) throw std::runtime_error("Fichier vide : " + path);
        int r = static_cast<int>(tmp.size());
        int c = static_cast<int>(tmp[0].size());
        Grid<T> g(r, c);
        for (int i = 0; i < r; ++i)
            for (int j = 0; j < c; ++j)
                g(i, j) = tmp[i][j];
        return g;
    }

    // Sauvegarde dans un fichier texte
    void save(const std::string& path) const {
        std::ofstream f(path);
        if (!f) throw std::runtime_error("Impossible d'écrire : " + path);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (j) f << ' ';
                f << static_cast<int>((*this)(i, j));
            }
            f << '\n';
        }
    }

    void print() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (j) std::cout << ' ';
                std::cout << std::setw(2) << static_cast<int>((*this)(i, j));
            }
            std::cout << '\n';
        }
    }
};
