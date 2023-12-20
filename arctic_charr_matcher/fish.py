class Fish:
    def __init__(
        self,
        image_path,
        cave_number,
        month,
        year,
        mask_path=None,
        spot_path=None,
        maskLabel=None,
        spotLabel=None,
        spotJson=None,
        precomp=None,
        precompAA=None,
    ):
        self._image_path = image_path
        self._cave_number = cave_number
        self._month = month
        self._year = year
        self._mask_path = mask_path
        self._spot_path = spot_path
        self._maskLabel = maskLabel
        self._spotLabel = spotLabel
        self._spotJson = spotJson
        self._precomp = precomp
        self._precompAA = precompAA

    def __str__(self):
        return f"Fish: {self.image_path}, Cave: {self.cave_number}, Mask: {self.mask_path}, Spot: {self.spot_path}"

    def __repr__(self):
        return f"Fish: {self.image_path}, Cave: {self.cave_number}, Mask: {self.mask_path}, Spot: {self.spot_path}"

    def __eq__(self, other):
        if isinstance(other, Fish):
            return self.image_path == other.image_path

    @property
    def uuid(self):
        imgFile = self._image_path.split("/")[-1].split(".")[0]
        return f"C{self._cave_number}-{self._year}-{self._month}-{imgFile}"

    @property
    def image_path(self):
        return self._image_path

    @image_path.setter
    def image_path(self, image_path):
        self._image_path = image_path

    @property
    def cave_number(self):
        return self._cave_number

    @cave_number.setter
    def cave_number(self, cave_number):
        self._cave_number = cave_number

    @property
    def month(self):
        return self._month

    @month.setter
    def month(self, month):
        self._month = month

    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, year):
        self._year = year

    @property
    def mask_path(self):
        return self._mask_path

    @mask_path.setter
    def mask_path(self, mask_path):
        self._mask_path = mask_path

    @property
    def spot_path(self):
        return self._spot_path

    @spot_path.setter
    def spot_path(self, spot_path):
        self._spot_path = spot_path

    @property
    def maskLabel(self):
        return self._maskLabel

    @maskLabel.setter
    def maskLabel(self, maskLabel):
        self._maskLabel = maskLabel

    @property
    def spotLabel(self):
        return self._spotLabel

    @spotLabel.setter
    def spotLabel(self, spotLabel):
        self._spotLabel = spotLabel

    @property
    def spotJson(self):
        return self._spotJson

    @spotJson.setter
    def spotJson(self, spotJson):
        self._spotJson = spotJson

    @property
    def precomp(self):
        return self._precomp

    @precomp.setter
    def precomp(self, precomp):
        self._precomp = precomp

    @property
    def precompAA(self):
        return self._precompAA

    @precompAA.setter
    def precompAA(self, precompAA):
        self._precompAA = precompAA
