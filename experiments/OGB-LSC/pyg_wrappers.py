from torch_sparse import SparseTensor


class DGraphSparseTensor(SparseTensor):
    def __init__(
        self,
        row,
        col,
        value=None,
        sparse_sizes=None,
        is_sorted=False,
        comm=None,
        rank_mapping=None,
        **kwargs,
    ):
        super(DGraphSparseTensor, self).__init__(
            row=row,
            col=col,
            value=value,
            sparse_sizes=sparse_sizes,
            is_sorted=is_sorted,
        )
        assert comm is not None, "Comm object cannot be None"
        assert rank_mapping is not None, "rank_mapping cannot be None"
        self.comm = comm
        self.rank_mapping = rank_mapping
        self.world_size = comm.get_world_size()
        self.rank = comm.get_rank()
