# API Layer 🚀

## Descripción
Esta capa actúa como el **único punto de entrada** para el mundo exterior. Su responsabilidad principal es exponer los endpoints del servicio, validar que las peticiones sean correctas y delegar la lógica de negocio a la capa de **Orchestration**.

## Responsabilidades
- **Exposición de Endpoints:** Definición de rutas REST para la ingesta de reseñas y generación de análisis FODA.
- **Validación de Datos (Schema Validation):** Asegurar que el input del usuario (IDs de apps, fechas, parámetros de filtrado) cumpla con el formato esperado antes de procesarlo.
- **Gestión de Respuestas:** Estandarizar el formato de salida (JSON) y los códigos de estado HTTP (200, 201, 400, 500, etc.).
- **Manejo de Errores:** Capturar excepciones de las capas internas y devolver mensajes de error amigables y seguros al cliente.
- **Seguridad y Control:** Gestión de autenticación (API Keys/JWT), límites de tasa (Rate Limiting) y logs de peticiones.

## Estructura de Contenido
- `routes/`: Definición de los endpoints (ej. `/analyze`, `/reviews`).
- `schemas/`: Modelos de validación de entrada y salida (ej. Pydantic Models).
- `middlewares/`: Filtros para autenticación, registro de métricas y manejo global de excepciones.
- `dependencies/`: Inyección de dependencias (instancias de servicios de orquestación).

## Interacciones
- **Hacia afuera:** Se comunica con clientes externos (Web, Mobile, Terceros).
- **Hacia adentro:** Invoca exclusivamente a la capa de **Orchestration**. No tiene conocimiento de la base de datos (DataAccess) ni de la configuración específica de los LLMs (Models).

---
*Nota: Esta capa debe mantenerse "delgada" (Thin Controller), evitando contener lógica de procesamiento o transformación de datos.*