# DS-fraud-example

Repositorio para análisis exploratorio, prueba de modelos y deployment de modelo ganador.

## Deployment

En test/test.py se encuentra un ejecutable ejemplo para probar la lambda function de AWS que genera predicciones para saber si un cliente es fraudulento o no en base a su transaccionalidad.

El deployment del modelo realiza lo siguiente:
 1. Genera un predicted en base a 13 features.
 2. Guarda el registro dentro de una tabla en Athena, para ello, se requiere el `user_id`.
 3. El json que necesita ingresarse es como el siguiente ejemplo:

```
{
    'user_id': 3938.0,
    'monto_count_Aceptada': 1.0,
    'diff_hours_max': np.nan,
    'cashback_min_En_proceso': 0.0,
    'num_ciudades': 1.0,
    'monto_mean_Aceptada': 134.86,
    'dispositivo_año_2018': 0.0,
    'cashback_max_Aceptada': 4.05,
    'establecimiento_Supermercado': 0.0,
    'interes_tc': 53.0,
    'establecimiento_Tienda_departamental': 1.0,
    'establecimiento_Compra_en_línea': 0.0,
    'cashback_mean_Rechazada': 0.0,
    'cashback_max_En_proceso': 0.0
}
```

## Contenido del Repositorio

```
├── README.md                          <- Documento de referencia para navegar el repositorio del proyecto
│
├── data                               <- Base de datos utilizada para realizar el proyecto (no recomendable cargar, 
|                                         sólo para fines de reproducibilidad); objeto modelo
│
├── dev                                <- Deployment de modelo para lambda en AWS usando contenedores
|   ├── fraud_model_predictions.py     <- Lambda function para cálculo de predicteds
|   ├── utils.py                       <- Funciones auxiliares para ejecución de app
|   ├── dockerfile                     <- Para creación de contenedor que será guardado dentro de AWS
|   ├── requirements.txt               <- Librerías de python necesarias para ejecución de app
│   ├── sql                            <- Queries para crear database y tabla donde las predicciones serán guardadas 
|
├── source                             <- Notebook con análisis exploratorio, prueba y creación de modelo
|
├── test                               <- Contiene un .py para test de lambda function con un caso de ejemplo
│
├── utils                              <- Funciones auxiliares utilizadas en notebook para creación de modelo
|
└── requirements.txt                   <- Librerías necesarias para reproducibilidad de notebook
```