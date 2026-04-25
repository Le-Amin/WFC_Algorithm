#!/bin/bash
#SBATCH --job-name=wfc_omp
#SBATCH --output=results/wfc_%j.out
#SBATCH --error=results/wfc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:15:00
#SBATCH --partition=cpu_short

set -e

echo "=== WFC Algorithm - Benchmark OpenMP ==="
echo "Date : $(date)"
echo "Noeud : $(hostname)"
echo "CPUs disponibles : $(nproc)"

# Compilation
echo ""
echo "--- Compilation ---"
make clean
make serial omp

# -------------------------------------------------------
# Benchmarks version série
# -------------------------------------------------------
echo ""
echo "--- Version série ---"
for SIZE in 16 32 64 128; do
    echo -n "Grille ${SIZE}x${SIZE} (binaire) : "
    ./wfc_serial samples/binary_5x5.txt $SIZE $SIZE serial 42
done
echo ""
for SIZE in 16 32 64; do
    echo -n "Grille ${SIZE}x${SIZE} (multi) : "
    ./wfc_serial samples/multi_8x8.txt $SIZE $SIZE serial 42
done

# -------------------------------------------------------
# Benchmarks version OpenMP (scalabilité en threads)
# -------------------------------------------------------
echo ""
echo "--- Version OpenMP (scalabilité) ---"
for THREADS in 1 2 4 8 16 32; do
    echo -n "  $THREADS thread(s) - grille 64x64 (binaire) : "
    OMP_NUM_THREADS=$THREADS ./wfc_omp samples/binary_5x5.txt 64 64 omp 42
done

echo ""
echo "--- Version OpenMP - grandes grilles (16 threads) ---"
for SIZE in 32 64 128 256; do
    echo -n "  ${SIZE}x${SIZE} (binaire) : "
    OMP_NUM_THREADS=16 ./wfc_omp samples/binary_5x5.txt $SIZE $SIZE omp 42
done

echo ""
echo "=== Terminé ==="
