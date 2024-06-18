from connect_to_db import connect_to_mongo

# This function adds a block to the database whenever theres a new block type
def add_block():
    """Add a block to the block_types collection."""

    # Connect to the database
    client = connect_to_mongo()
    
    # Connect to the block_types collection
    db = client.ampli_cv
    collection = db.block_types
    
    # Insert documents
    documents = [
        {"block_name": "Wick Block", "Sequence": (1,1,1,3)},
        {"block_name": "Sample Block", "Sequence": (2,2,1,3)},
        {"block_name": "Conjugate Pad", "Sequence": (3,3,3,1)},
        {"block_name": "Test Block 1", "Sequence": (2,2,1,2)},
        {"block_name": "Test Block 2", "Sequence": (1,1,2,1)},
        {"block_name": "Test Block 3", "Sequence": (3,3,2,3)},
        {"block_name": "Control Block", "Sequence": (2,2,3,2)},
    ]   
    collection.insert_many(documents)


    client.close()

def delete_all():
    """Delete all documents in the block_types collection."""
    
    # Connect to the database
    client = connect_to_mongo()
    
    # Connect to the block_types collection
    db = client.ampli_cv
    collection = db.block_types

    # Delete all documents
    collection.delete_many  ({})

    # Close the connection
    client.close()

    print("All documents deleted.")

def get_all():
    """Get all documents in the block_types collection."""
    
    # Connect to the database
    client = connect_to_mongo()
    
    # Connect to the block_types collection
    db = client.ampli_cv
    collection = db.block_types

    # Get all documents
    cursor = collection.find({})
    for document in cursor:
        print(document)

    # Close the connection
    client.close()

if __name__ == "__main__":
    delete_all()
    add_block()
    get_all()