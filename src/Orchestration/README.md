# Orchestration Layer 🧠

## Descripción
Esta capa actúa como el **cerebro o director de orquesta** de la aplicación. Su función es coordinar el flujo de datos entre la capa de **DataAccess** y la de **Models** para ejecutar los casos de uso del negocio, como la generación de un análisis FODA.

## Responsabilidades
- **Control del Workflow:** Define la secuencia lógica de pasos (ej: extraer reseñas → limpiar texto → enviar al LLM → guardar resultado).
- **Gestión de Contexto:** Selecciona y prepara la información relevante (contexto) que se enviará al LLM.
- **Manejo de Prompt Templates:** Recupera las plantillas de instrucciones y las completa con los datos dinámicos de las reseñas.
- **Lógica de Negocio:** Toma decisiones basadas en el estado del proceso (ej: si hay pocas reseñas, decidir si abortar o continuar el análisis).
- **Transformación de Salida:** Mapea la respuesta cruda del modelo al formato estructurado final (Matriz FODA) requerido por la API.

## Estructura de Contenido
- `services/`: Clases o módulos que implementan los casos de uso (ej. `FodaAnalysisService`).
- `chains/`: Lógicas de pasos múltiples (si se usan frameworks como LangChain). **(a consultar)**
- `prompts/`: Gestión de las plantillas de instrucciones para el análisis.**(a consultar)**
- `evaluators/`: Lógica para verificar la calidad o coherencia de la respuesta generada por el modelo. **(a consultar)**

## Interacciones
- **Hacia arriba:** Recibe instrucciones de la **API Layer**.
- **Hacia abajo:** 
    - Solicita datos y persiste resultados en **DataAccess**.
    - Envía prompts y recibe inferencias de **Models**.
- **Independencia:** No conoce detalles de transporte (HTTP) ni de infraestructura de base de datos; solo maneja la lógica procedimental.

---
*Nota: Esta capa garantiza que el sistema sea agnóstico al modelo. Si se cambia de GPT-4 a Llama-3, la lógica de orquestación debería permanecer prácticamente intacta.*
