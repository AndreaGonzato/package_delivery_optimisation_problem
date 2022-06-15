from customer import Customer


class DoorToDoorCustomer(Customer):

    door_to_door_customer_index = 0  # this is a static value for the class

    def __new__(cls, location, package_demand=1):
        obj = object.__new__(cls)
        return obj

    def __init__(self, location, package_demand=1):
        super().__init__(location, package_demand)
        self.door_to_door_customer_index = DoorToDoorCustomer.door_to_door_customer_index
        DoorToDoorCustomer.door_to_door_customer_index += 1

    def __repr__(self):
        return "CD" + str(self.door_to_door_customer_index)
