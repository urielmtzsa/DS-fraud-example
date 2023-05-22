import io
import boto3
import pandas as pd
import pickle as pkl
import joblib
    
def read_s3(bucket,path,filename):
    s3 = boto3.client("s3")
    ext = filename.split(".")[-1]
    if ext in ["csv"]:
        with io.BytesIO() as f:
            s3.download_fileobj(bucket,path+filename,f)
            f.seek(0)
            file = pd.read_csv(f)
    elif ext in ["pkl","pickle"]:
        with io.BytesIO() as f:
            s3.download_fileobj(bucket,path+filename,f)
            f.seek(0)
            file = pkl.load(f)
    elif ext in ["joblib"]:
        with io.BytesIO() as f:
            s3.download_fileobj(bucket,path+filename,f)
            f.seek(0)
            file = joblib.load(f)  
    else:
        raise ValueError(f"'{ext}' extension is not supported")
    return file


def write_athena(query,database = "pred",output = 's3://bucket-ums/athena/queries/'):
    
    client = boto3.client('athena')
    response = client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': database},
        ResultConfiguration={'OutputLocation': output}
    )
    return None