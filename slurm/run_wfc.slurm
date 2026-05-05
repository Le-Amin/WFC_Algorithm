#!/bin/bash
#SBATCH --job-name=wfc_omp
#SBATCH --output=results/wfc_%j.out
#SBATCH --error=results/wfc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --account=r250127
#SBATCH --constraint=armgpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G

set -e

mkdir -p results

echo "=== WFC Algorithm - Benchmark OpenMP ==="
echo "Date        : $(date)"
echo "Noeud       : $(hostname)"
echo "CPUs alloués: ${SLURM_CPUS_PER_TASK}"

# ── Compilation ────────────────────────────────────────────────────────────────
echo ""
echo "--- Compilation ---"
make clean
make serial omp
echo "Compilation OK"

# ── Utilitaire de mesure (répète 3x, affiche la médiane) ──────────────────────
run() {
    local label="$1"; shift
    local times=()
    for i in 1 2 3; do
        t=$( { time "$@" > /dev/null; } 2>&1 | grep real | awk '{print $2}' \
             | sed 's/m/:/;s/s//' | awk -F: '{printf "%.3f", $1*60+$2}' )
        times+=("$t")
    done
    # tri et médiane manuelle (bash pur)
    IFS=$'\n' sorted=($(sort -n <<<"${times[*]}")); unset IFS
    echo "${label} : ${sorted[1]} s  (x3 médiane)"
}

# ── 1. Version série — référence ──────────────────────────────────────────────
echo ""
echo "--- Référence série ---"
for SIZE in 16 32 64 128 256; do
    run "serial  binary ${SIZE}x${SIZE}" \
        ./wfc_serial samples/binary_5x5.txt $SIZE $SIZE serial 42
done
for SIZE in 16 32 64 128; do
    run "serial  multi  ${SIZE}x${SIZE}" \
        ./wfc_serial samples/multi_8x8.txt  $SIZE $SIZE serial 42
done

# ── 2. Scalabilité en threads (grille 128×128) ────────────────────────────────
echo ""
echo "--- Scalabilité threads (binary 128x128) ---"
for T in 1 2 4 8 16 32; do
    run "omp T=${T}  binary 128x128" \
        env OMP_NUM_THREADS=$T ./wfc_omp samples/binary_5x5.txt 128 128 omp 42
done

# ── 3. Scalabilité en taille (16 threads) ────────────────────────────────────
echo ""
echo "--- Scalabilité taille (16 threads, binaire) ---"
for SIZE in 32 64 128 256 512; do
    run "omp T=16 binary ${SIZE}x${SIZE}" \
        env OMP_NUM_THREADS=16 ./wfc_omp samples/binary_5x5.txt $SIZE $SIZE omp 42
done

echo ""
echo "--- Scalabilité taille (16 threads, multi) ---"
for SIZE in 32 64 128 256; do
    run "omp T=16 multi  ${SIZE}x${SIZE}" \
        env OMP_NUM_THREADS=16 ./wfc_omp samples/multi_8x8.txt $SIZE $SIZE omp 42
done

echo ""
echo "=== Terminé : $(date) ==="
