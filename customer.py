class Customer:

    def __new__(cls, index, location, package_demand=1):
        obj = object.__new__(cls)
        return obj

    def __init__(self, index, location, package_demand=1):
        self.index = index
        self.location = location
        self.package_demand = package_demand

    def __repr__(self):
        return "C" + str(self.index)

    def get_nearest_store(self, stores):
        min_distance = float("inf")
        nearest_store = stores[0]
        for store in stores:
            distance = self.location.euclidean_distance(store.location)
            if distance < min_distance:
                min_distance = distance
                nearest_store = store
        return nearest_store
