
class Utils:
    # Translates the x,y coordinates to the equivalent index of grid_ds.
    @staticmethod
    def xy_to_index(Grid , x:int , y:int):
        """
        ### XY to index
        ---------------
        Function that translates the x,y coordinates to the equivalent index of grid_ds.
        
        #### Args:
        * x: x coordinate of the point
        * y: y coordinate of the point
        * Grid: Grid object

        #### Returns:
        * index of the point in the grid_ds
        """

        x_index = int(round(x // Grid.SQUARE_LENGTH))
        y_index = int(round(y // Grid.SQUARE_LENGTH))

        return (min(x_index, Grid.MAX_INDEX), min(y_index, Grid.MAX_INDEX))

    # Same as above but taking into consideration the skewing that happens near the outter squares.
    @classmethod
    def xy_to_index_skewed(cls, Grid, x: int, y: int, a:float):

        middle_index_tl_x = (Grid.SQUARE_LENGTH * Grid.MAX_INDEX)/2
        middle_index_tl_y = (Grid.SQUARE_LENGTH * Grid.MAX_INDEX)/2
        
        index_x, index_y = cls.xy_to_index(Grid, x , y)

        offset_x = int(abs(middle_index_tl_x - index_x)**a)
        offset_y = int(abs(middle_index_tl_y - index_y)**a)

        index_x_skewed = int(round(x // (Grid.SQUARE_LENGTH + offset_x)))
        index_y_skewed = int(round(y // (Grid.SQUARE_LENGTH + offset_y)))

        return (min(index_x_skewed, Grid.MAX_INDEX), min(index_y_skewed, Grid.MAX_INDEX))

    # Translates the index to the equivalent x,y coordinates of grid_ds top left point.   
    @staticmethod
    def index_to_xy(Grid, x_index:int, y_index:int):
        """
        ### Index to XY
        ---------------
        Function that translates the index to the equivalent x,y coordinates of grid_ds tl point.

        #### Args:
        * x_index: x index of the point
        * y_index: y index of the point
        * grid_ds: Grid object

        #### Returns:
        * x,y coordinates of the top left point of the square
        """
    
        x = (x_index) * Grid.SQUARE_LENGTH
        y = (y_index) * Grid.SQUARE_LENGTH

        return (x, y) # top left point of square