class BackendEngine(object):
    def __init__(self):
        pass

    def init_process_group(self, *args, **kwargs):
        raise NotImplementedError

    def get_rank(self):
        raise NotImplementedError

    def get_world_size(self):
        raise NotImplementedError

    def scatter(self, *args, **kwargs):
        raise NotImplementedError

    def gather(self, *args, **kwargs):
        raise NotImplementedError
