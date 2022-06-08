from location import Location
from locker import Locker

if __name__ == "__main__":

    l1 = Location(0, 0)
    l2 = Location(2, 2)
    print(l1.euclidean_distance(l2))

