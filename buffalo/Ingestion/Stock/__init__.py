from .PolygonStockGrepper import *

class StockGrepper(PolygonStockGrepper):
    """
    This Grepper provides endpoint of access of stocks from users. 
    """
    def __init__(self, **init_args) -> None:
        super().__init__(**init_args)

    