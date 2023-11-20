from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from io import BytesIO
import pypdf
import os



blob_service_client = BlobServiceClient.from_connection_string(os.environ['API_KEY'])
clientContainer = os.environ['CONTAINER']


def get_blob(blob_name = "MasakhaNEWS_News_Topic_Classification_for_African_.pdf"):
  
    blob_client = blob_service_client.get_blob_client(container=os.environ["CONTAINER"], blob=blob_name)

    streamdownloader = blob_client.download_blob() # download blob stream
    with open(file=os.path.join(r'utils', 'document.pdf'), mode="wb") as sample_blob:
        download_stream =streamdownloader
        sample_blob.write(download_stream.readall())