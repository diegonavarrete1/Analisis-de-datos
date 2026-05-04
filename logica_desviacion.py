import math
import random


# Cálculo de la desviacion estandar
def desviacion_estandar(datos):
    N = len(datos)
    if(N == 0):
        print("No se tienen datos para calcular el desviacion estandar")
    else:
        miu = 0
        for x in datos:
            miu += x
        miu = miu/N
        suma_dif = 0
        for x_i in datos:
            # abs obtiene el valor absoluto de un numero, el operador ** eleva a la potencia dada un numero
            suma_dif += abs(x_i - miu) ** 2

        de = math.sqrt(suma_dif/N)
        return de 

# creamos una lista con 100 datos pseudoaleatorios
print("desviacion estandar de 100 numeros enteros aleatorios en el rango [0, 1000]")
datos = []
for i in range(100):
    d_i = random.randint(0,1000)
    #print(i, d_i)
    datos.append(d_i)
mi_de = desviacion_estandar(datos)
print(mi_de)
print(mi_de + 100)

# creamos una lista con 1000 datos pseudoaleatorios
print("desviacion estandar de 10000 numeros enteros aleatorios en el rango [0, 1000]")
datos2 = []
for i in range(10000):
    datos2.append(random.randint(0, 1000))

print(desviacion_estandar(datos2))




# Funcion que genera una cadena pseudoaleatoria de longitud n
def mono(n):
    caracteres = "abcdefghijklmnñopqrstuvwxyz 1234567890"
    cadena_resultado = ""
    for i in range(n):
        indice = random.randint(0, len(caracteres)-1)
        cadena_resultado = cadena_resultado + caracteres[indice]
    return cadena_resultado



print(mono(10))
print(mono(20))




