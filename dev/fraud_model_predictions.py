from utils import read_s3,write_athena
import pandas as pd
import numpy as np
from datetime import datetime

def lambda_handler(event, context):
    nan = np.nan
    NaN = np.nan
    none = np.nan
    metadata = [event["queryStringParameters"]]
    bucket = "bucket-ums"
    path = "model-ds-fraud/"
    filename = "model.pkl"
    database = "pred"
    table = "fraud"
    model = read_s3(bucket,path,filename)
    features = list(model["select"].feature_names_in_)
    df = pd.DataFrame(metadata)
    df = df.astype(float)
    df["predicted"] = model.predict_proba(df)[:,1]
    results = df[["user_id","predicted"]].set_index("user_id").to_dict()
    t = datetime.today() 
    t = t.strftime("%Y-%m-%d %H:%M:%S")
    df["created_at"] = t
    df = df[["user_id"]+features+["created_at"]].copy()
    query = f"INSERT INTO {database}.{table} VALUES {str([tuple(x) for x in df.values])[1:-1]}"
    query = query.replace(f"'{t}'",f"CAST('{t}' AS TIMESTAMP)")
    qresponse = write_athena(query)
    return {
        "statusCode": 200,
        "body": results
    }

