class Customer:
    counter = 0  # this is a static value for the class

    def __new__(cls, location):
        obj = object.__new__(cls)
        return obj

    def __init__(self, location, package_demand=1):
        self.index = Customer.counter
        self.location = location
        self.package_demand = package_demand
        Customer.counter += 1

    def __repr__(self):
        return "{index: " + str(self.index) + ", location: " + str(self.location) + "}"
