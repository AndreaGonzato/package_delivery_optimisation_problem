from location import Location
from locker import Locker
from customer import Customer

if __name__ == "__main__":
    l = Locker(Location(0, 7), 70)
    print(l.capacity)

