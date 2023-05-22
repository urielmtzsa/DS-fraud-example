import numpy as np
import requests

url = "https://johntmpjeclqqdiyhbwvtrvmqm0woeiz.lambda-url.us-east-1.on.aws/"

inpt  ={'user_id': 3938.0,
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
 'cashback_max_En_proceso': 0.0}

response = requests.get(url, params= inpt)
print(response.json())