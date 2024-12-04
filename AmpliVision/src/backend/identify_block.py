import warnings


""" 
This is our \'database\' of color ranges and block types.
We previously used MongoDB to store this information, 
but its so little that we can just hardcode it.
"""
color_ranges = [
    {"color_name": "Red", "color#" : 1, "min": (150,0,0), "max": (255,150,149)},
    {"color_name": "Red", "color#" : 1, "min": (100,0,0), "max": (149,99,149)},
    {"color_name": "Blue", "color#" : 2, "min": (0,0,150), "max": (255,168,255)},
    {"color_name": "Green", "color#" : 3, "min": (0,100,0), "max": (149,255,149)},
    {"color_name": "Green", "color#" : 3, "min": (150,151,0), "max": (255,255,149)}
]

block_types = [
    {"block_name": "wick_block", "Sequence": (1, 1, 1, 3)},
    {"block_name": "sample_block", "Sequence": (2, 2, 1, 3)},
    {"block_name": "conjugate_pad", "Sequence": (3, 3, 3, 1)},
    {"block_name": "test_block1", "Sequence": (2, 2, 1, 2)},
    {"block_name": "test_block2", "Sequence": (1, 1, 2, 1)},
    {"block_name": "test_block3", "Sequence": (3, 3, 2, 3)},
    {"block_name": "control_block", "Sequence": (2, 2, 3, 2)},
]

def rgb_to_number(rgb, color_ranges):
    """Convert an RGB value to a number using the local color_ranges."""
    r, g, b = rgb

    for color_range in color_ranges:
        if (color_range['min'][0] <= r <= color_range['max'][0] and
            color_range['min'][1] <= g <= color_range['max'][1] and
            color_range['min'][2] <= b <= color_range['max'][2]):
            return color_range['color#']

    warnings.warn(f"RGB sequence '{r},{g},{b}' not found in color ranges.")
    return None

def identify_block(block, display: int = 0):
    """Function to identify the block type of a block given the RGB sequence."""

    # Get the RGB sequence of the block
    sequence_rgb = []
    sequence_numerical = []
    for rgb in block.get_rgb_sequence():
        sequence_rgb.append(rgb)
        number = rgb_to_number(rgb, color_ranges)
        if number is not None:
            sequence_numerical.append(number)

    # Print the sequences if display is set to 1
    if display:
        print(f'RGB sequence: {sequence_rgb}')
        print(f'Numerical sequence: {sequence_numerical}')

    # Check if the sequence is in the block_types list
    for rotation in range(len(sequence_numerical)):
        query_sequence = tuple(sequence_numerical)

        for block_type in block_types:
            if block_type['Sequence'] == query_sequence:
                block.block_type = block_type["block_name"]
                if display:
                    print(f'\'{block.block_type}\' at {block.index}\n')

                r = [0, 90, 180, 270]
                block.rotation = r[rotation]
                return block

        # Rotate the sequence for the next iteration
        sequence_numerical = sequence_numerical[1:] + sequence_numerical[:1]

    # If the sequence is not found, warn and return the block as unknown
    warnings.warn(f'\nBlock: Unknown at {block.index}. #Seq {sequence_numerical} \n')
    
    return block
    
if __name__ == '__main__':
    ...
