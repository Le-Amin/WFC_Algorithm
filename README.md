# Wave Function Collapse — CHPS0801

Implémentation de l'algorithme **Wave Function Collapse** (modèle overlapping)  
en C++17 avec versions série et parallèle (OpenMP tasks).

## Structure

```
include/
  grid.hpp          Grille 2D générique Grid<T>
  tile.hpp          Tuile carrée Tile<N,T> + règles de compatibilité
  wfc.hpp           Algorithme WFC (extraction tuiles, propagation, résolution)
src/
  main.cpp          Point d'entrée (parse args, chronomètre, sauvegarde)
  wfc_serial.hpp    Résolution série (inclus via wfc.hpp)
  wfc_omp.hpp       Résolution parallèle OpenMP tasks (inclus via wfc.hpp)
samples/
  binary_5x5.txt    Exemple de l'énoncé (grille {0,1} 5×5)
  multi_8x8.txt     Grille multi-valeurs {0,1,2} 8×8
slurm/
  run_wfc.slurm     Script SLURM pour Romeo (benchmark complet)
results/            Sorties générées
```

## Compilation

```bash
make serial    # binaire sans OpenMP
make omp       # binaire avec OpenMP
make all       # les deux
make clean
```

## Utilisation

```
./wfc_serial <sample> <rows> <cols> [serial|omp] [seed]
./wfc_omp    <sample> <rows> <cols> [serial|omp] [seed]
```

Exemples :
```bash
./wfc_serial samples/binary_5x5.txt 16 16 serial 42
OMP_NUM_THREADS=8 ./wfc_omp samples/multi_8x8.txt 32 32 omp 42
```

## Sur Romeo (HPC)

```bash
salloc -N1 --cpus-per-task=32 -p cpu_short
srun --cpus-per-task=32 bash slurm/run_wfc.slurm
# ou directement :
sbatch slurm/run_wfc.slurm
```

## Algorithme

1. **Extraction des tuiles** : sous-grilles N×N de l'échantillon S (avec fréquences)
2. **Compatibilité** : `compatible[t][dy][dx]` = tuiles pouvant se chevaucher avec t à l'offset (dx,dy)
3. **Wave** : pour chaque cellule, ensemble de tuiles encore possibles
4. **Boucle principale** :
   - Choisir la cellule de plus faible entropie
   - Collapse aléatoire pondéré par fréquence
   - Propagation des contraintes (AC-3 simplifié)

## Parallélisation OpenMP

- **Recherche du minimum d'entropie** : `#pragma omp parallel for`
- **Propagation** : chaque voisin à mettre à jour est soumis comme `#pragma omp task`
- Synchronisation par `#pragma omp critical` et verrous `omp_lock_t`
