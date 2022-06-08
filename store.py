class Store:
    counter = 0  # this is a static value for the class

    def __new__(cls, location, capacity):
        obj = object.__new__(cls)
        return obj

    def __init__(self, location, capacity, is_warehouse=False):
        self.index = Store.counter
        if self.index == 0:
            self.is_warehouse = True
        self.location = location
        self.capacity = capacity
        Store.counter += 1

    def __repr__(self):
        return "{index: " + str(self.index) + ", location: " + str(self.location) + "}"


