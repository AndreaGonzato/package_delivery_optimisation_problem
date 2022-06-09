from vehicle_type import VehicleType


class Vehicle:
    counter = 0  # this is a static value for the class

    def __new__(cls, vehicle_type, departure_store, capacity):
        obj = object.__new__(cls)
        return obj

    def __init__(self, vehicle_type, departure_store, capacity):
        self.index = Vehicle.counter
        self.vehicle_type = vehicle_type
        self.departure_store = departure_store
        self.capacity = capacity
        Vehicle.counter += 1

    def __repr__(self):
        return "{index: " + str(self.index) + ", vehicle_type: " + str(self.vehicle_type) + ", departure_store: " + \
               str(self.departure_store) + ", capacity: " + str(self.capacity) + "}"