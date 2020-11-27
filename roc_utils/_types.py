class StructContainer():
    """
    Build a type that behaves similar to a struct.

    Usage:
        # Construction from named arguments.
        settings = StructContainer(option1 = False,
                                   option2 = True)
        # Construction from dictionary.
        settings = StructContainer({"option1": False,
                                    "option2": True})
        print(settings.option1)
        settings.option2 = False
        for k,v in settings.items():
            print(k,v)
    """

    def __init__(self, dictionary=None, **kwargs):
        if dictionary is not None:
            assert(isinstance(dictionary, (dict, StructContainer)))
            self.__dict__.update(dictionary)
        self.__dict__.update(kwargs)

    def __iter__(self):
        for i in self.__dict__:
            yield i

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __len__(self):
        return sum(1 for k in self.keys())

    def __repr__(self):
        return "struct(**%s)" % str(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def items(self):
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                yield (k, v)

    def keys(self):
        for k in self.__dict__:
            if not k.startswith("_"):
                yield k

    def values(self):
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                yield v

    def update(self, data):
        self.__dict__.update(data)

    def asdict(self):
        return dict(self.items())

    def first(self):
        # Assumption: __dict__ is ordered (python>=3.6).
        key, value = next(self.items())
        return key, value

    def last(self):
        # Assumption: __dict__ is ordered (python>=3.6).
        # See also: https://stackoverflow.com/questions/58413076
        key = list(self.keys())[-1]
        return key, self[key]

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def setdefault(self, key, default=None):
        return self.__dict__.setdefault(key, default)
