import boto3
import sys
if sys.version[0]=='2':
    from io import BytesIO as StringIO # for python 2.7
else:
    from io import StringIO
DESTINATION = 'XXXX_your_s3_destination_bucket_XXXX'

def _write_dataframe_to_csv_on_s3(dataframe, filename):
    """ Write a dataframe to a CSV on S3 """
    print("Writing {} records to {}".format(len(dataframe), filename))
    # Create buffer
    csv_buffer = StringIO()
    # Write dataframe to buffer
    dataframe.to_csv(csv_buffer, sep="|", index=False)
    # Create S3 object
    s3_resource = boto3.resource("s3")
    # Write buffer to S3 object
    s3_resource.Object(DESTINATION, filename).put(Body=csv_buffer.getvalue())
    
bucket_name='XXXX_your_s3_bucket_name_XXXX'
file_path = 'XXXX_s3_file_path_XXXXX'
import boto3
from io import StringIO,BytesIO

def dataframe_to_s3( input_datafame, bucket_name, filepath, file_format='csv'):
    s3 = boto3.client("s3")

    if file_format == 'parquet':
        out_buffer = BytesIO()
        input_datafame.to_parquet(out_buffer, index=False)

    elif file_format == 'csv':
        out_buffer = StringIO()
        input_datafame.to_csv(out_buffer, index=False)

    s3.put_object(Bucket=bucket_name, Key=filepath, Body=out_buffer.getvalue())