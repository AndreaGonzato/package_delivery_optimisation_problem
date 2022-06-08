class Locker:

    index = 0
    x = 0
    y = 0

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __int__(self, idx, x_coordinate, y_coordinate):
        self.index = idx
        self.x = x_coordinate
        self.y = y_coordinate


    def __repr__(self) -> str:
        return "{index: "+str(self.index)+", x: "+str(self.x)+", y: "+str(self.y)+"}"

    def print_locker(self):
        print("hi from Locker class")


