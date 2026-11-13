import torch


class _CommaString(object):
    """ A full string separated by commas. """
    def __init__(self, text: str):
        self.text = text
        return

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text

    def has_next_comma(self) -> bool:
        return ',' in self.text

    def _read_next(self) -> str:
        """
        :return: the raw string of next token before comma
        """
        if self.has_next_comma():
            token, self.text = self.text.split(',', maxsplit=1)
        else:
            token, self.text = self.text, ''
        return token.strip()

    def read_next_as_int(self) -> int:
        return int(self._read_next())

    def read_next_as_float(self) -> float:
        return float(self._read_next())

    def read_next_as_bool(self) -> bool:
        """ Parse the next token before comma as boolean, 1/0 for true/false. """
        num = self.read_next_as_int()
        assert num == 1 or num == 0, f'The should-be-bool number is {num}.'
        return bool(num)
    pass


def load_nnet(filepath: str):
    """ Load from dumped file in NNET format.
    :return: Tuple of <AcasNet, input mins vector, input maxs vector>
    """
    # ===== Basic Initializations =====
    _num_layers = 0  # Number of layers in the network (excluding inputs, hidden + output = num_layers).
    _input_size = 0  # Number of inputs to the network.
    _output_size = 0  # Number of outputs to the network.
    _max_layer_size = 0  # Maximum size dimension of a layer in the network.

    _layer_sizes = []  # Array of the dimensions of the layers in the network.

    _symmetric = False  # Network is symmetric or not. (was 1/0 for true/false)

    _mins = []  # Minimum value of inputs.
    _maxs = []  # Maximum value of inputs.

    _means = []  # Array of the means used to scale the inputs and outputs.
    _ranges = []  # Array of the ranges used to scale the inputs and outputs.

    _layer_weights = []  # holding concrete weights of each layer
    _layer_biases = []  # holding concrete biases of each layer

    # ===== Now loading from files =====

    with open(filepath, 'r') as f:
        line = f.readline()
        while line.startswith('//'):
            # ignore first several comment lines
            line = f.readline()

        # === Line 1: Basics ===
        data = _CommaString(line)
        _num_layers = data.read_next_as_int()
        _input_size = data.read_next_as_int()
        _output_size = data.read_next_as_int()
        _max_layer_size = data.read_next_as_int()
        print('_num_layers', _num_layers)
        print('_input_size', _input_size)
        print('_output_size', _output_size)
        print('_max_layer_size', _max_layer_size)
        
        
        # === Line 2: Layer sizes ===
        data = _CommaString(f.readline())
        for _ in range(_num_layers + 1):
            _layer_sizes.append(data.read_next_as_int())

        assert _layer_sizes[0] == _input_size
        assert _layer_sizes[-1] == _output_size
        assert all(size <= _max_layer_size for size in _layer_sizes)
        assert len(_layer_sizes) >= 2, f'Loaded layer sizes have {len(_layer_sizes)} (< 2) elements?! Too few.'

        # === Line 3: Symmetric ===
        data = _CommaString(f.readline())
        _symmetric = data.read_next_as_bool()
        assert _symmetric is False, "We don't know what symmetric==True means."

        # It has to read by line, because in following lines, I noticed some files having more values than needed..

        # === Line 4: Mins of input ===
        data = _CommaString(f.readline())
        for _ in range(_input_size):
            _mins.append(data.read_next_as_float())

        # === Line 5: Maxs of input ===
        data = _CommaString(f.readline())
        for _ in range(_input_size):
            _maxs.append(data.read_next_as_float())

        # === Line 6: Means ===
        data = _CommaString(f.readline())
        # the [-1] is storing the size for output normalization
        for _ in range(_input_size + 1):
            _means.append(data.read_next_as_float())

        # === Line 7: Ranges ===
        data = _CommaString(f.readline())
        # the [-1] is storing the size for output normalization
        for _ in range(_input_size + 1):
            _ranges.append(data.read_next_as_float())

        # === The rest are layer weights/biases. ===
        for k in range(_num_layers):
            in_size = _layer_sizes[k]
            out_size = _layer_sizes[k + 1]

            # read "weights"
            tmp = []
            for i in range(out_size):
                row = []
                data = _CommaString(f.readline())
                for j in range(in_size):
                    row.append(data.read_next_as_float())
                tmp.append(row)
                assert not data.has_next_comma()

            """ To fully comply with NNET in Reluplex, DoubleTensor is necessary.
                Otherwise it may record 0.613717 as 0.6137170195579529.
                But to make everything easy in PyTorch, I am just using FloatTensor.
            """
            _layer_weights.append(torch.tensor(tmp))

            # read "biases"
            tmp = []
            for i in range(out_size):
                # only 1 item for each
                data = _CommaString(f.readline())
                tmp.append(data.read_next_as_float())
                assert not data.has_next_comma()

            _layer_biases.append(torch.tensor(tmp))
            pass

        data = _CommaString(f.read())
        assert not data.has_next_comma()  # should have no more data
    print(len(_layer_weights))
    for i in _layer_weights:
        print(i.shape)
    # ===== Use the parsed information to build AcasNet =====
    _hidden_sizes = _layer_sizes[1:-1]  # exclude inputs and outputs sizes
    from network import FNN
    acas = FNN()
    para = {}
    l = 0
    for key in acas.state_dict().keys():
        if 'weight' in key:
            para[key] = _layer_weights[l]
        else:
            para[key] = _layer_biases[l]
            l += 1
    acas.load_state_dict(para)
    p = path.find('run2a_')
    q = path.find('batch')
    r = path[p + 6: q].replace('_', '')
    
    name = '/data/home/mjnn/majianan/ProvRepair/model/n' + r + '.pth'
    torch.save(acas.state_dict(), name)
    # net = AcasNet(dom, _input_size, _output_size, _hidden_sizes, _means, _ranges)

    # # === populate weights and biases ===
    # assert len(net.all_linears) == len(_layer_weights) == len(_layer_biases)
    # for i, linear in enumerate(net.all_linears):
    #     linear.weight.data = _layer_weights[i]
    #     linear.bias.data = _layer_biases[i]


    # return net, _mins, _maxs

# for i in range(2, 6):
#     for j in range(1, 10):
#         path = f'/data/home/mjnn/majianan/abLocal/safety/model/ACASXU_run2a_{i}_{j}_batch_2000.nnet'
#         load_nnet(path)
