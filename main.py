from location import Location
from locker import Locker
from customer import Customer
from locker_customer import LockerCustomer
from door_to_door_customer import DoorToDoorCustomer

if __name__ == "__main__":
    l = Locker(Location(5, 5), 60)
    print(l)

    #lc = LockerCustomer(Location(0, 7), l, package_demand=3)
    #print(lc)

    dc = DoorToDoorCustomer(Location(99, 1), prime=True)
    print(dc)

