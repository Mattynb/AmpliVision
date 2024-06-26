from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

try:
    from secrets_ import URI
except:
    from .secrets_ import URI


def connect_to_mongo():
    """
    Connect to MongoDB Atlas database.
    """    

    # Connect to your Atlas cluster
    client = MongoClient(URI, server_api=ServerApi('1'))
    
    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        #print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)

    return client


if __name__ == "__main__":
    connect_to_mongo()