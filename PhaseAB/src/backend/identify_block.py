import warnings
from .add_to_db.connect_to_db import Client

def identify_block(block, display: int = 0):
    """ Function to identify the block type of a block given the RGB sequence."""

    client = Client

    # Connect to the color_ranges database collection
    db = client.ampli_cv
    collection = db.color_ranges

    # Get the RGB sequence of the
    # block in rgb and numerical form
    sequence_rgb = []
    sequence_numerical = []
    for rgb in block.get_rgb_sequence():
        # Add the RGB to the sequence_rgb list
        sequence_rgb.append(rgb)

        # Convert the RGB to a number and
        # add the number to the sequence_numerical list
        number = rgb_to_number(rgb, collection)
        sequence_numerical.append(number)

    # Print the sequences
    if display:
        print(f'RGB sequence: {sequence_rgb}')
        print(f'Numerical sequence: {sequence_numerical}')

    # Connect to the block_types collection
    block_collection = db.block_types

    # Check if the sequence is in the database
    # If it is, print the block type
    for rotation in range(len(sequence_numerical)):
        # Look for the sequence in the database
        query = {'Sequence': sequence_numerical}

        # If the sequence is found, print the block type and return
        block_type = block_collection.find_one(query)
        if block_type:
            block.block_type = block_type["block_name"]
            if display:
                print(f'\'{block_type["block_name"]}\' at {block.index}\n')


            r = [0, 90, 180, 270]
            block.rotation = r[rotation]

            return block

        # Rotate the sequence
        sequence_numerical = sequence_numerical[1:] + sequence_numerical[:1]

    # If the sequence is not found, print unknown
    warnings.warn(f'\nBlock: Unknown at {block.index}. #Seq {sequence_numerical} \n')

    return block


def rgb_to_number(rgb, collection):
    """ 
    Convert an RGB value to a number using the color_ranges collection in the database.
    """
    r, g, b = rgb

    query = {
        # Compare Red range
        'min.0': {'$lte': r},
        'max.0': {'$gte': r},

        # Compare Green range
        'min.1': {'$lte': g},
        'max.1': {'$gte': g},

        # Compare Blue range
        'min.2': {'$lte': b},
        'max.2': {'$gte': b},
    }

    # Find the color number
    numbers = collection.find(query)
    numbers = [number['color#'] for number in numbers ]
    
    # If sequence not found
    if len(numbers) == 0:
        warnings.warn(f"RGB sequence \'{r},{g},{b}\' not in database")

    # If there are multiple colors, print the colors and return the first color
    # Note that there should not be multiple colors for a single RGB value
    if len(numbers) > 1:
        warnings.warn(f"Multiple colors found for r: {r}, g: {g}, b: {b}...\n{[number for number in numbers]}")

    # If there is only one color, return the color number
    return numbers[0]

    
if __name__ == '__main__':
    ...
