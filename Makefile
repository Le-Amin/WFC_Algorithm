CXX      := g++
# -static évite les DLL manquantes sous Windows/MinGW
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Iinclude -static
OMPFLAGS := -fopenmp

SRC      := src/main.cpp
TARGET   := wfc

.PHONY: all serial omp clean

all: serial omp

# Binaire série (sans OpenMP)
serial: $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET)_serial $(SRC)

# Binaire parallèle (avec OpenMP)
omp: $(SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DWFC_OMP -o $(TARGET)_omp $(SRC)

# Test rapide sur l'exemple de l'énoncé
test_serial: serial
	./$(TARGET)_serial samples/binary_5x5.txt 10 10 serial 42

test_omp: omp
	OMP_NUM_THREADS=4 ./$(TARGET)_omp samples/binary_5x5.txt 10 10 omp 42

test_multi: omp
	OMP_NUM_THREADS=4 ./$(TARGET)_omp samples/multi_8x8.txt 16 16 omp 42

clean:
	rm -f $(TARGET)_serial $(TARGET)_omp results/*.txt
