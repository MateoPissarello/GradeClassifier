# Explicación del script `q_learning_policy.py`

Este documento describe, paso a paso, cómo está construido el simulador de Q-Learning para el proyecto **GradeClassifier**. La idea es que puedas modificarlo o exponerlo con total claridad durante el taller.

## 1. Propósito general

El script crea un entorno de aprendizaje por refuerzo muy compacto que representa a los cuatro clústeres de estudiantes obtenidos en el análisis exploratorio. Cada clúster se trata como un **estado** y cada intervención académica posible como una **acción**. Mediante Q-Learning, el agente aprende qué intervención aplicar para mover al estudiante hacia estados con mejor rendimiento académico.

## 2. Estructuras principales del código

### 2.1 `StateProfile`
Una dataclass que encapsula la información descriptiva de cada clúster: etiqueta, promedio de GPA, promedio de ausencias y un resumen textual. Esta información se usa para calcular recompensas y generar explicaciones legibles.

### 2.2 `AcademicInterventionEnv`
Representa el entorno de refuerzo tabular. Sus partes clave son:

- **`state_profiles`**: diccionario con los cuatro clústeres descritos en el cuaderno original (por ejemplo, "Cluster 0 – Bajo estudio + más ausencias").
- **`actions`**: lista de intervenciones académicas modeladas como acciones (`intensive_tutoring`, `attendance_monitoring`, `balanced_support`, `wellbeing_program`).
- **`transition_model`**: reglas heurísticas que especifican a qué clúster se mueve el estudiante después de aplicar cada acción, junto con sus probabilidades.
- **`initial_distribution`**: distribución inicial aproximada de estudiantes por clúster; se usa en `reset()` para seleccionar el estado de partida de cada episodio.
- **Métodos de interacción**:
  - `reset()`: elige un estado inicial según la distribución anterior.
  - `sample_transition()`: ejecuta la transición probabilística al siguiente estado.
  - `reward()`: calcula la recompensa comparando el GPA y las ausencias entre el estado actual y el siguiente.
  - `step()`: combinación de transición y recompensa, tal como se espera en un entorno de RL.

## 3. Algoritmo de Q-Learning

La función `q_learning()` aplica Q-Learning estándar con epsilon-greedy y decaimiento lineal de `epsilon`:

1. Inicializa una tabla `Q` con ceros (una fila por estado y una columna por acción).
2. Para cada episodio, elige un estado inicial con `reset()`.
3. Mientras no se alcance el máximo de pasos, selecciona una acción:
   - Explora con probabilidad `epsilon`.
   - Aprovecha la mejor acción según la fila de `Q` con probabilidad `1 - epsilon`.
4. Observa el siguiente estado y la recompensa, y actualiza la tabla siguiendo la ecuación clásica de Q-Learning.
5. Al final, devuelve la tabla `Q` completa.

Los hiperparámetros (`episodes`, `alpha`, `gamma`, etc.) son modificables desde la llamada a la función.

## 4. Derivación y evaluación de la política

- `greedy_policy()`: transforma la tabla `Q` en una política determinista eligiendo, para cada estado, la acción con mayor valor esperado.
- `simulate_policy()`: evalúa la política usando simulaciones Monte Carlo. Calcula la recompensa promedio por paso y la proporción de transiciones que logran una mejora de GPA.
- `describe_policy()`: genera un reporte amigable (en español) explicando la acción recomendada en cada clúster.

## 5. Ejecución del script

Al ejecutar `python q_learning_policy.py`:

1. Se fija una semilla aleatoria para obtener resultados reproducibles.
2. Se crea el entorno, se entrena el agente, se obtiene la política óptima (codificada en la tabla `Q`).
3. Se imprime la tabla `Q`, la explicación de la política y métricas de desempeño promedio.

Este flujo permite experimentar con los parámetros del algoritmo o ajustar el modelo de transiciones y recompensas para analizar distintos escenarios educativos.

## 6. Cómo adaptar el código

- **Cambiar clústeres o atributos**: modifica `state_profiles` para reflejar nuevos grupos de estudiantes.
- **Probar nuevas intervenciones**: agrega acciones en `self.actions` y define sus transiciones dentro de `transition_model`.
- **Redefinir la recompensa**: ajusta la función `reward()` para priorizar objetivos distintos (por ejemplo, disminuir ausencias más agresivamente).
- **Explorar hiperparámetros**: experimenta con más episodios, tasas de aprendizaje diferentes o políticas de exploración alternativas.

Con estos elementos, tendrás una visión completa de cómo funciona el simulador y cómo llevarlo al taller.
