TARGET = prog
NVCC   = nvcc

# Flags de compilación
# -O3           : optimización
# -Xcompiler    : pasa flags al compilador C++ (gcc)
# -fopenmp      : habilita OpenMP en el host
NVCC_FLAGS = -O3 -Xcompiler -fopenmp

SRC = main.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
