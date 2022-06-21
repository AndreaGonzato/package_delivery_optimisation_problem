from customer import Customer


class DoorToDoorCustomer(Customer):

    def __new__(cls, index, door_to_door_customer_index, location, package_demand=1, prime=False):
        obj = object.__new__(cls)
        return obj

    def __init__(self, index, door_to_door_customer_index, location, package_demand=1, prime=False):
        super().__init__(index, location, package_demand)
        self.prime = prime
        self.door_to_door_customer_index = door_to_door_customer_index

    def __repr__(self):
        return "CD" + str(self.door_to_door_customer_index)

    def set_prime(self, prime):
        self.prime = prime
