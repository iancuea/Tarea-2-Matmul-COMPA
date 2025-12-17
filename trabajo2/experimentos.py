import subprocess
import matplotlib.pyplot as plt
import numpy as np

# ---------------- CONFIGURACIÓN ----------------
ns = [512, 1024, 2048]
hilos = 4
repeticiones = 5
algs = {1: "CPU", 
        2: "GPU",  
        3: "GPUSm"}

resultados = {name: [] for name in algs.values()}

print(f"Iniciando experimentos (promediando {repeticiones} ejecuciones)...")

# ---------------- EXPERIMENTOS ----------------
for n in ns:
    for aid, nombre in algs.items():
        tiempos_corrida = []

        for _ in range(repeticiones):
            res = subprocess.run(
                ["./prog", str(n), str(hilos), str(aid)],
                capture_output=True,
                text=True
            )

            salida = res.stdout.strip()

            try:
                # Buscar la palabra "Tiempo"
                for linea in salida.splitlines():
                    if "Tiempo" in linea:
                        # Ejemplo: "Tiempo GPU: 0.0123 s"
                        tiempo = float(linea.split(":")[1].replace("s", "").strip())
                        tiempos_corrida.append(tiempo)
                        break
            except:
                print(f"Error leyendo tiempo: N={n}, Alg={nombre}")

        promedio = sum(tiempos_corrida) / len(tiempos_corrida)
        resultados[nombre].append(promedio)

        print(f"N={n:5} | Alg={nombre:5} | Tiempo Promedio = {promedio:.6f} s")

# ---------------- GRÁFICO TIEMPO ----------------
plt.figure(figsize=(10, 6))
for nombre in algs.values():
    plt.plot(ns, resultados[nombre], marker='o', label=nombre)

plt.xlabel("Tamaño de la Matriz (N)")
plt.ylabel("Tiempo promedio (s)")
plt.title("Tiempo de ejecución vs Tamaño de matriz")
plt.legend()
plt.grid(True)
plt.savefig("tiempo.png")

# ---------------- GRÁFICO SPEEDUP ----------------
plt.figure(figsize=(10, 6))

speedup_gpu = np.array(resultados["CPU"]) / np.array(resultados["GPU"])
speedup_sm  = np.array(resultados["CPU"]) / np.array(resultados["GPUSm"])

plt.plot(ns, speedup_gpu, marker='s', label="Speedup GPU Básica")
plt.plot(ns, speedup_sm, marker='^', label="Speedup GPU Shared")

plt.xlabel("Tamaño de la Matriz (N)")
plt.ylabel("Speedup (CPU / GPU)")
plt.title("Speedup respecto a CPU")
plt.legend()
plt.grid(True)
plt.savefig("speedup.png")

print("\nListo. Gráficos guardados como 'tiempo.png' y 'speedup.png'.")
