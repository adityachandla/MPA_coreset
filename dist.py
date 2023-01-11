import math

def dist_square(p1: [float,float], p2: [float,float]) -> float:
    return (p2[0]-p1[0])**2 + (p2[1]-p1[1])**2

def dist(p1: [float,float], p2: [float,float]) -> float:
    return math.sqrt(dist_square(p1, p2))
