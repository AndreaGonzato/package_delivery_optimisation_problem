class Location:

    x = 0
    y = 0

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __int__(self, x_coordinate, y_coordinate):
        self.x = x_coordinate
        self.y = y_coordinate

    def __repr__(self) -> str:
        return "{x: "+str(self.x)+", y: "+str(self.y)+"}"
