from customer import Customer


class LockerCustomer(Customer):

    def __new__(cls, index, locker_customer_index, location, store, package_demand=1):
        obj = object.__new__(cls)
        return obj

    def __init__(self, index, locker_customer_index, location, store, package_demand=1):
        super().__init__(index, location, package_demand)
        self.store = store
        self.locker_customer_index = locker_customer_index
        self.did_not_show_up = False

    def __repr__(self):
        return "CL" + str(self.locker_customer_index)

    def set_did_not_show_up(self, did_not_show_up):
        self.did_not_show_up = did_not_show_up
