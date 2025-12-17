# Tarea 2: Multiplicación de Matrices (Matmul) - Computación Paralela
Integrantes: Marcelo Lara e Ian Cuevas
Este proyecto implementa y compara tres versiones del algoritmo de multiplicación de matrices (Matmul) para analizar el rendimiento entre CPU y GPU (CUDA).

## 1. Hardware Utilizado
Para obtener estos datos, ejecuta `lscpu` y `nvidia-smi` en tu terminal y completa los campos:
- **CPU:** AMD Ryzen 7 4000 Series (8 cores / 16 threads)
- **GPU:** NVIDIA GeForce RTX 2060 (6 GB VRAM)
- **Memoria RAM:** 16 GB
- **Sistema Operativo:** Ubuntu (Linux)
- **Cuda Toolkit** CUDA 12.0

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
Esto generara los archivos `tiempo.png` y  `speedup.png`
## 4. Análisis de Resultados 

### Comparación de Tiempos
Al observar el gráfico `tiempo.png`, se puede notar lo siguiente:

* **CPU:**
    * El tiempo de ejecución aumenta rápidamente a medida que crece el tamaño de la matriz ($N$).
    * Esto se debe a la complejidad cúbica del algoritmo de multiplicación de matrices, $O(n^3)$, y a la cantidad limitada de núcleos disponibles en la CPU para procesar tal carga.
* **GPU:**
    * Las versiones en GPU presentan tiempos mucho menores que la CPU, especialmente para matrices grandes.
    * Para tamaños pequeños, la diferencia no es tan marcada debido al *overhead* inicial de ejecución en GPU.
* **Nota sobre la medición:**
    * En el caso de la GPU, el tiempo medido en este reporte corresponde a la ejecución del kernel CUDA, excluyendo las transferencias de memoria para aislar el rendimiento del cómputo.

### Análisis de Speedup
A partir del gráfico `speedup.png`:

* **GPU vs CPU:**
    * Se observa un *speedup* significativo que aumenta a medida que el tamaño de la matriz crece.
    * Esto demuestra que la arquitectura de la GPU aprovecha de manera superior el paralelismo masivo para este tipo de tareas.
* **GPU con Memoria Compartida:**
    * La versión con memoria compartida (`GPUSm`) supera a la GPU básica para matrices grandes.
    * Esto ocurre porque la memoria compartida tiene una latencia mucho menor que la memoria global y permite reutilizar datos dentro de cada bloque de hilos, reduciendo el tráfico de memoria.

## 5. Conclusiones
Los resultados obtenidos muestran que la paralelización en GPU entrega mejoras de rendimiento críticas frente a la ejecución tradicional en CPU. Mientras la CPU está limitada por su número reducido de núcleos, la GPU puede ejecutar miles de hilos en paralelo, reduciendo drásticamente los tiempos de ejecución. 

Finalmente, el uso de **memoria compartida** en CUDA se valida como una optimización clave para operaciones intensivas, logrando una eficiencia superior a la versión básica al optimizar el acceso a los datos.
