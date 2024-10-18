import torch
import numpy

if __name__ == '__main__':
    data = numpy.linspace(1, 10, 10).reshape(2, 5)
    tensor = torch.from_numpy(data)
    print(data)
    print(tensor)
    print(tensor.__hash__())

    tensor = torch.linspace(1, 10, 10).reshape(2, 5)

    tensor2 = torch.normal(0, 1, size=(2, 5))
    print(tensor2)

    tensor3 = torch.normal(tensor, 1)
    print(tensor3)

    tensor4 = torch.normal(0, tensor)
    print(tensor4)

    tensor5 = torch.normal(tensor, tensor)
    print(tensor5)
