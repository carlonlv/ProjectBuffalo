import enum


class Storage(enum.Enum):
    SQLITE = "sqlite"
    CSV = "csv"
    EXCEL = "xlsx"
    PICKLE = "pkl"


class API(enum.Enum):
    POLYGON = enum.auto()
