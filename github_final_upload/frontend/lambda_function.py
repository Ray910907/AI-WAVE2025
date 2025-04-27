import os
import io
import boto3
import json
import csv

# Grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    
    # Expecting request as a python dictionary with {"data" : sample} format
    payload = event['data']
    print(payload)
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       Body=payload,
                                       ContentType="text/csv"
                                       )
    print(f'res:{response}')
    result = response["Body"].read().decode("utf-8")
    print(result)
    pred = float(result)
    predicted_label = {'riskScore': float(result)}
    return predicted_label