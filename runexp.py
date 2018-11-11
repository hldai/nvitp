from reader import TextReader
from models.nvdm import NVDM
from models.gsmlda import GSMLDA


if __name__ == '__main__':
    dataset = 'ptb'
    data_path = 'd:/data/cdcr/nvitp/{}'.format(dataset)
    reader = TextReader(data_path)
    # model = NVDM(reader, dataset)
    # model.train()
    model = GSMLDA(reader)
    model.train()
