from connect_to_db import connect_to_mongo

# This function adds a block to the database whenever theres a new block type
def add_block():
    """Add a block to the block_types collection."""

    # Connect to the database
    client = connect_to_mongo()
    
    # Connect to the block_types collection
    db = client.ampli_cv
    collection = db.block_types

    # Insert a document
    post = {"block_name": "Wick Block", "Sequence": (1,1,1,3)}  # Sequence is tl, tr, bl, br
    post_id = collection.insert_one(post).inserted_id
    print(post_id)
    post = {"block_name": "Sample Block", "Sequence": (2,2,1,3)}
    post_id = collection.insert_one(post).inserted_id
    print(post_id)
    post = {"block_name": "Conjugate Pad", "Sequence": (3,3,3,1)}
    post_id = collection.insert_one(post).inserted_id
    print(post_id)
    post = {"block_name": "Test Block", "Sequence": (2,2,1,2)}
    post_id = collection.insert_one(post).inserted_id
    print(post_id)
    post = {"block_name": "Control Block", "Sequence": (2,2,3,2)}
    post_id = collection.insert_one(post).inserted_id
    print(post_id)
    # Close the connection
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