from vehicle_type import VehicleType


class Vehicle:

    def __new__(cls, index, vehicle_type, departure_store, capacity):
        obj = object.__new__(cls)
        return obj

    def __init__(self, index, vehicle_type, departure_store, capacity):
        self.index = index
        self.vehicle_type = vehicle_type
        self.departure_store = departure_store
        self.capacity = capacity

    def __repr__(self):
        return "{index: " + str(self.index) + ", vehicle_type: " + str(self.vehicle_type) + ", departure_store: " + \
               str(self.departure_store) + ", capacity: " + str(self.capacity) + "}"
