from FAParser.tasks.parser_task import ParserTask


class ConstituencyParser(ParserTask):
    def __init__(self, args):
        return super().__init__(args)

    def load_dataset(self, split, combine=False, **kwargs):
        data_set = self.dataset[split] # Tree File
        return super().load_dataset(split, combine=combine, **kwargs)
