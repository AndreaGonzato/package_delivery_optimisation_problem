class Store:
    counter = 0  # this is a static value for the class

    def __new__(cls, location, capacity, is_warehouse=False):
        obj = object.__new__(cls)
        return obj

    def __init__(self, location, capacity, is_warehouse=False):
        self.index = Store.counter
        self.location = location
        self.capacity = capacity
        if self.index == 0 or is_warehouse:
            self.is_warehouse = True
        else:
            self.is_warehouse = False
        Store.counter += 1

    def __repr__(self):
        return "{index: " + str(self.index) + ", location: " + str(self.location) + "}"


