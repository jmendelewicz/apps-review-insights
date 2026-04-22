ADR 0001 — <Data Ingestion Library>
Fecha: 2026-03-10
Estado: Aprobado
Autores: <Matias Alberdi>
Issue/PR: (https://trello.com/c/tV8eWlh6/9-discovery-proveedor-de-rese%C3%B1as)

##### Contexto
Se debe decidir que medio se empleará en pos de obtener las reseñas para alimentar el modelo. Generar un scrapper para un proveedor determinado sería costoso en términos de tiempo. La obtención de las reseñas acota la flexibilidad del producto y la accesibilidad a las mismas puede impactar en la reproducibilidad de los experimentos o la necesidad de almacenar mayor cantidad de datos.

##### Decisión
Decidimos usar la librería GooglePlayScrapper puesto que nos ahorra tiempo de desarrollo, nos permite acceder a reseñas de aplicaciones variopintas, su código es de libre acceso y no requiere autenticacion ni API key alguna.

***Alternativas consideradas***
Se planteó como alternativa utilizar al API de Steam para acceder a reseñas de juegos.

**Pros**
La API de Steam provee reseñas de forma abierta y directa, sin límites de descarga.

**Contras**
Dichas reseñas estan acotadas a los juegos que ofrece la plataforma, las mismas NO cuentan con una valoración numérica. 

##### Consecuencias
Se pretende que esta librería acote los tiempos de desarrollo, pero no ate el proyecto a un proveedor puntual de las mismas (esto debe ser posibilitado desde el diseño/arquitectura de la aplicación).
Desafortunadamente, GooglePlayScrapper no cuenta con filtros por fechas por lo cual, para ganar reproducibilidad de los experimentos, se deberán almacenar las reseñas sin procesar en lugar de sólo los parámetros que permitan su obtención.

Ejemplo:
“Ganamos realismo en evaluación. Perdemos tamaño de train. Riesgo: si hay estacionalidad fuerte, el test puede ser más difícil.”

Cómo se valida esta decisión
Definí “cómo sabemos que estuvo bien”:

Métrica/criterio esperado: …
Comparación con baseline: …
Chequeos (ej: leakage): …

##### Implementación

1. Escribir un módulo dentro de **src/DataAccess/Ingestion** que envuelva la implementacion de la librería.
2. Llamar dicho módulo desde la capa de **Orchestration**, para que coordine la descarga de las reseñas y almacenamiento de los datos llamando al módulo correspondiente dentro de **DataAccess/Storage**.

Puntualmente, se utilizará el método *reviews* siguiendo las sugerencias provistas por la documentacion de la librería

```python
batch, token = reviews(
                app_id,
                lang=self.lang,
                country=self.country,
                sort=sort,
                count=min(200, remaining),
                continuation_token=token
            )
```

Esto retorna una lista de diccionarios, compuestos por los siguientes campos
* reviewId
* userName
* userImage
* content
* score
* thumbsUpCount
* reviewCreatedVersion
* at
* replyContent
* repliedAt
* appVersion

De los cuales sólo utilizaremos los siguientes.

| Campo | Descripción |
|---|---|
| `content` | Texto de la reseña |
| `score` | Puntaje del usuario (1–5 estrellas) |
| `thumbsUpCount` | Cantidad de likes de la reseña |
| `reviewCreatedVersion` | Versión de la app al momento de la reseña |
| `at` | Fecha de la reseña |

Para este sprint, los campos utilizados son `content` (texto) y `score`.

Para más información sobre la librería, visitar https://pypi.org/project/google-play-scraper/

##### Prueba rápida

El siguiente script, descarga el conjunto de los últimos 1000 reseñas de whatsapp y guarda el resultado dentro de la carpeta ***data***

```python
scrapper = GooglePlayScraper(lang="es", country="ar")
results = scrapper.get_reviews(app_id="com.whatsapp", limit=1000)
CSVReviewRepository().save(pd.DataFrame(results), "data/reviews.csv")
```