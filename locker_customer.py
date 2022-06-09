from customer import Customer


class LockerCustomer(Customer):
    locker_customer_index = 0  # this is a static value for the class

    def __new__(cls, location, store, package_demand=1):
        obj = object.__new__(cls)
        return obj

    def __init__(self, location, store, package_demand=1):
        super().__init__(location, package_demand)
        self.store = store
        self.locker_customer_index = LockerCustomer.locker_customer_index
        LockerCustomer.locker_customer_index += 1

    def __repr__(self):
        return "CL" + str(self.locker_customer_index)
