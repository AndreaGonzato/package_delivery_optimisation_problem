from locker_customer import LockerCustomer


class Store:
    counter = 0  # this is a static value for the class
    stores = []

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
        Store.stores.append(self)

    def __repr__(self):
        if self.is_warehouse:
            return "W"
        else:
            return "L" + str(self.index)

    def find_associated_CL(self, customers, stores):
        associated_locker_customer = []
        if not self.is_warehouse:
            # only for locker
            C_L = list(filter(lambda customer: type(customer) == LockerCustomer, customers))
            for cl in C_L:
                if cl.get_nearest_store(stores).index == self.index:
                    associated_locker_customer.append(cl)
        return associated_locker_customer






