import math

class Location:
    counter = 0

    def __new__(cls, x, y):
        obj = object.__new__(cls)
        return obj

    def __init__(self, x, y):
        self.index = Location.counter
        self.x = x
        self.y = y
        Location.counter += 1

    def __repr__(self):
        return "{x: " + str(self.x) + " y: " + str(self.y) + "}"

    def euclidean_distance(self, other_location):
        delta_x = self.x - other_location.x
        delt_y = self.y - other_location.y
        return math.sqrt(delta_x*delta_x + delt_y*delt_y)
