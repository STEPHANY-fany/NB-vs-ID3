import pandas as pd
from collections import Counter
import math

file_path = r"C:\Users\bamao\Desktop\Codigos\conjunto_de_datos_divocios_2018.csv"
data = pd.read_csv(file_path)
data = data[['ini_juic', 'causa', 'hijos', 'postr_div1', 'postr_div2', 'hij_men', 'custodia']].dropna()

def calcular_entropia(datos, columna_objetivo):
    valores, conteos = Counter(datos[columna_objetivo]).keys(), Counter(datos[columna_objetivo]).values()
    total = sum(conteos)
    return -sum((c / total) * math.log2(c / total) for c in conteos)

def ganancia_informacion(datos, columna, columna_objetivo):
    entropia_inicial = calcular_entropia(datos, columna_objetivo)
    valores = datos[columna].unique()
    total = len(datos)
    entropia_condicional = sum(
        (len(datos[datos[columna] == valor]) / total) * 
        calcular_entropia(datos[datos[columna] == valor], columna_objetivo)
        for valor in valores
    )
    return entropia_inicial - entropia_condicional

def construir_arbol(datos, columnas, columna_objetivo):
    if len(datos[columna_objetivo].unique()) == 1:
        return datos[columna_objetivo].iloc[0]
    if not columnas:
        return datos[columna_objetivo].mode()[0]
    mejor_columna = max(columnas, key=lambda col: ganancia_informacion(datos, col, columna_objetivo))
    arbol = {mejor_columna: {}}
    for valor in datos[mejor_columna].unique():
        subconjunto = datos[datos[mejor_columna] == valor]
        arbol[mejor_columna][valor] = construir_arbol(subconjunto, [col for col in columnas if col != mejor_columna], columna_objetivo)
    return arbol

def predecir(arbol, ejemplo):
    if not isinstance(arbol, dict):
        return arbol
    columna = next(iter(arbol))
    valor = ejemplo.get(columna, None)
    if valor not in arbol[columna]:
        return None
    return predecir(arbol[columna][valor], ejemplo)

def traducir_resultado(resultado):
    if resultado == 0:
        return "La custodia fue otorgada al padre."
    elif resultado == 1:
        return "La custodia fue otorgada a la madre."
    elif resultado == 2:
        return "La custodia fue compartida."
    else:
        return "Resultado desconocido."

columnas_predictoras = ['ini_juic', 'causa', 'hijos', 'postr_div1', 'postr_div2', 'hij_men']
columna_objetivo = 'custodia'
arbol_decision = construir_arbol(data, columnas_predictoras, columna_objetivo)

ejemplo = {
    'ini_juic': 2,
    'causa': 28,
    'hijos': 1,
    'postr_div1': 1,
    'postr_div2': 1,
    'hij_men': 0  # 'FALSO' se representa como 0
}
resultado = predecir(arbol_decision, ejemplo)
mensaje = traducir_resultado(resultado)
print(f"Predicción: {mensaje}")

