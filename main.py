from location import Location
from store import Store
from customer import Customer
from locker_customer import LockerCustomer
from door_to_door_customer import DoorToDoorCustomer

if __name__ == "__main__":
    l1 = Store(Location(0, 0), 60)
    l2 = Store(Location(40, 50), 60)
    l3 = Store(Location(100, 100), 60)
    stores = [l1, l2, l3]
    c = Customer(Location(50, 50))
    print(c.get_nearest_store(stores))

    #lc = LockerCustomer(Location(0, 7), l, package_demand=3)
    #print(lc)

    #dc = DoorToDoorCustomer(Location(99, 1), prime=True)
    #print(dc)

