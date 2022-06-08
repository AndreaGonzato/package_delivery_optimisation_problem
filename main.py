from location import Location
from locker import Locker
from customer import Customer
from locker_customer import LockerCustomer

if __name__ == "__main__":
    l = Locker(Location(5, 5), 60)
    print(l)
    lc = LockerCustomer(Location(0, 7), l)
    print(lc)

