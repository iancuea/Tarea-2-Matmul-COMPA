TARGET = prog
NVCC   = nvcc

NVCC_FLAGS = -O3 -Xcompiler -fopenmp

SRC = main.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
