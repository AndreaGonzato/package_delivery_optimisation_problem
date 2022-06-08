class Locker:
    counter = 0

    def __new__(cls, location):
        obj = object.__new__(cls)
        return obj

    def __init__(self, location):
        self.index = Locker.counter
        self.location = location
        Locker.counter += 1

    def __repr__(self):
        return "{index: " + str(self.index) + ", location: " + str(self.location) + "}"


