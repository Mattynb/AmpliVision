import os
from dotenv import load_dotenv

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


def connect_to_mongo():
    """
    Connect to MongoDB Atlas database.
    """

    # Load the URI from the environment
    load_dotenv()
    URI = os.getenv('URI')

    # DELETE this at some point
    URI = r'mongodb+srv://matheusberbet001:12345@amplicluster.0k26okc.mongodb.net/?retryWrites=true&w=majority&appName=AmpliCluster'
    print(URI)

    # Connect to your Atlas cluster
    client = MongoClient(URI, server_api=ServerApi('1'))

    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)

    return client

# Global var
Client = connect_to_mongo()

if __name__ == "__main__":
    connect_to_mongo()
