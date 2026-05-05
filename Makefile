CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Iinclude
OMPFLAGS := -fopenmp

SRC      := src/main.cpp
TARGET   := wfc

.PHONY: all serial omp clean test_serial test_omp test_multi

all: serial omp

results/:
	mkdir -p results

# Binaire série (sans OpenMP)
serial: $(SRC) | results/
	$(CXX) $(CXXFLAGS) -o $(TARGET)_serial $(SRC)

# Binaire parallèle (avec OpenMP)
omp: $(SRC) | results/
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $(TARGET)_omp $(SRC)

test_serial: serial
	./$(TARGET)_serial samples/binary_5x5.txt 10 10 serial 42

test_omp: omp
	OMP_NUM_THREADS=4 ./$(TARGET)_omp samples/binary_5x5.txt 10 10 omp 42

test_multi: omp
	OMP_NUM_THREADS=4 ./$(TARGET)_omp samples/multi_8x8.txt 16 16 omp 42

clean:
	rm -f $(TARGET)_serial $(TARGET)_omp
	rm -f results/*.txt results/*.out results/*.err
