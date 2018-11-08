import pickle as pkl

__all__ = [
    "get_collection",
    "add_to_collection",
    "save"
]


class Collections(object):
    """Collections for logs during training.

    Usually we add loss and valid metrics to some collections after some steps.
    """
    _MY_COLLECTIONS_NAME = "my_collections"

    def __init__(self, kv_stores=None, name=None):

        self._kv_stores = kv_stores if kv_stores is not None else {}

        if name is None:
            name = Collections._MY_COLLECTIONS_NAME
        self._name = name

    def load(self, archives):

        if self._name in archives:
            self._kv_stores = archives[self._name]
        else:
            self._kv_stores = []

    def add_to_collection(self, key, value):
        """
        Add value to collection

        :type key: str
        :param key: Key of the collection

        :param value: The value which is appended to the collection
        """
        if key not in self._kv_stores:
            self._kv_stores[key] = [value]
        else:
            self._kv_stores[key].append(value)

    def export(self):
        return {self._name: self._kv_stores}

    def get_collection(self, key):
        """
        Get the collection given a key

        :type key: str
        :param key: Key of the collection
        """
        if key not in self._kv_stores:
            return None
        else:
            return self._kv_stores[key]

    @staticmethod
    def pickle(path, **kwargs):
        """
        :type path: str
        """
        archives_ = dict([(k, v) for k, v in kwargs.items()])

        if not path.endswith(".pkl"):
            path = path + ".pkl"

        with open(path, 'wb') as f:
            pkl.dump(archives_, f)

    @staticmethod
    def unpickle(path):
        """:type path: str"""

        if not path.endswith(".pkl"):
            path = path + ".pkl"

        with open(path, 'rb') as f:
            archives_ = pkl.load(f)

        return archives_


# init global collection
_global_collection = Collections()


def get_collection(key, default=None):
    value = _global_collection.get_collection(key)

    if value is None:
        return [default] if default is not None else []

    return _global_collection.get_collection(key)


def add_to_collection(key, value):
    _global_collection.add_to_collection(key, value)


def save(path):
    archives = _global_collection.export()
    Collections.pickle(path, **archives)


def load(path):
    _global_collection.load(Collections.unpickle(path))
