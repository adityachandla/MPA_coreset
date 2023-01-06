from typing import NamedTuple

class RepresentativePoint(NamedTuple):
    coords: tuple[float,float]
    weight: int

class Square(NamedTuple): 
    topLeft: tuple[float,float]
    bottomRight: tuple[float, float]
