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

## Generacion de Graficos
Para ejecutar los experimentos automáticos y generar las comparativas:
```bash
python3 experimentos.py>
```
## 4. Análisis de Resultados 

### Comparativa de Tiempos
Al observar el archivo `tiempo.png`, se nota que:
* **Crecimiento en CPU:** La CPU presenta un crecimiento exponencial de tiempo a medida que $N$ aumenta, debido a la limitación de núcleos físicos frente a la carga computacional $O(n^3)$.
* **Rendimiento GPU:** Las versiones de GPU mantienen tiempos drásticamente más bajos. Sin embargo, para valores de $N$ muy pequeños (ej. 128 o 256), la CPU puede ser competitiva debido al *overhead* que implica la transferencia de datos del Host (CPU) al Device (GPU) y la inicialización del contexto de CUDA.

### Análisis de Speedup
Basado en `speedup.png`:
* **GPU vs CPU:** Se observa un factor de aceleración significativo (Speedup). A mayor tamaño de matriz, la GPU es más eficiente ya que utiliza miles de núcleos para procesar las operaciones de forma masivamente paralela.
* **Memoria Compartida (Shared Memory):** El algoritmo 3 (`GPUSm`) supera a la GPU básica en matrices grandes. Esto ocurre porque la memoria compartida tiene una latencia mucho menor que la memoria global. Al cargar bloques (*tiles*) de la matriz en la memoria local del bloque de hilos, se optimiza el reúso de datos y se evita saturar el ancho de banda de la memoria de video lenta.

## 5. Conclusiones
La paralelización en GPU es indispensable para el cálculo científico moderno. Mientras que la CPU actúa como un procesador generalista, la arquitectura paralela de CUDA permite reducir tiempos de procesamiento de minutos a milisegundos en tareas críticas de álgebra lineal, como la multiplicación de matrices.
