# INFO188_AyudantiaParalelismoCuda
Ayudantia para la asignatura INFO188 UACh sobre programacion paralela utilizando CUDA.
Ejemplos:
- Game of life example using CUDA
- Example of reduction sum of array using warp shuffle 

# Requisitos:
- Librerias basicas de c++
- Tener instalado CUDA

# Ejecucion
Ejecutar el makefile
```bash
make
```
Para ejecutar el juego de la vida ./prog
```bash
./prog <gpu-id>  n seed pasos <block-size>
```
Para ejecutar el warp shuffle ./prog
```bash
./prog <gpu-id>  n seed
```
Si seed = 0 el arreglo se construye solo con 1s
