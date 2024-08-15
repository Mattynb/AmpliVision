from connect_to_db import connect_to_mongo

def add_color_range():
    """ Add color ranges to the database color_ranges collection."""
    
    # Connect to the database
    client = connect_to_mongo()
    
    # Get the database and collection
    db = client.ampli_cv
    collection = db.color_ranges

    # Insert a document
    post = [
        {"color_name": "Red", "color#" : 1, "min": (150,0,0), "max": (255,150,149)},
        {"color_name": "Red", "color#" : 1, "min": (100,0,0), "max": (149,99,149)},
        {"color_name": "Blue", "color#" : 2, "min": (0,0,150), "max": (255+25,78+40+50,255)},
        {"color_name": "Green", "color#" : 3, "min": (0,100,0), "max": (149,255,149)},
        {"color_name": "Green", "color#" : 3, "min": (151,151,0), "max": (255,255,149)},
    ]

    # Insert the document
    post_id = collection.insert_many(post).inserted_ids
    print(post_id)

    # Close the connection
    client.close()

def delete_all():
    # Connect to the database
    client = connect_to_mongo()
    
    #  Get the database and collection
    db = client.ampli_cv
    collection = db.color_ranges

    # Delete all documents
    collection.delete_many({})

    # Close the connection
    client.close()

if __name__ == "__main__":
    delete_all()
    add_color_range()