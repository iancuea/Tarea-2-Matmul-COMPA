# Tarea 2: Multiplicación de Matrices (Matmul) - Computación Paralela

Este proyecto implementa y compara tres versiones del algoritmo de multiplicación de matrices (Matmul) para analizar el rendimiento entre CPU y GPU (CUDA).

## 1. Hardware Utilizado
Para obtener estos datos, ejecuta `lscpu` y `nvidia-smi` en tu terminal y completa los campos:
- **CPU:** [Escribe aquí el modelo, ej: Intel Core i7]
- **GPU:** [Escribe aquí el modelo, ej: NVIDIA RTX 3060]
- **Memoria RAM:** [Ej: 16 GB]
- **Sistema Operativo:** Ubuntu (Linux)

## 2. Algoritmos Implementados
1. **CPU Multicore (OpenMP):** Implementación en C++ usando hilos para paralelizar los bucles anidados.
2. **GPU Básica (CUDA):** Uso de memoria global. Cada hilo de la GPU calcula un elemento de la matriz resultante.
3. **GPU Shared Memory (CUDA):** Optimización que utiliza la memoria compartida (*tiles*) para reducir el tráfico de datos con la memoria global, mejorando la eficiencia en matrices grandes.

## 3. Instrucciones de Uso

### Compilación
El proyecto incluye un `Makefile` para facilitar la compilación:
```bash
make
```
## Ejecucion Manual
El programa acepta tres parámetros: n (tamaño), nt (hilos CPU) y alg (algoritmo 1, 2 o 3).
```bash
./prog <n> <nt> <alg>
```
##Generacion de Graficos
Para ejecutar los experimentos automáticos y generar las comparativas:
```bash
python3 experimentos.py>
```
