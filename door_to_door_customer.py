from customer import Customer


class DoorToDoorCustomer(Customer):

    def __new__(cls, location, package_demand=1, prime=False):
        obj = object.__new__(cls)
        return obj

    def __init__(self, location, package_demand=1, prime=False):
        super().__init__(location, package_demand)
        self.prime = prime

    def __repr__(self):
        return "{index: " + str(self.index) + ", location: " + str(self.location) + ", prime: " + str(self.prime) + "}"
