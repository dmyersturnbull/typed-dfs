from typing import Sequence
from typeddfs import SimpleFrame, OrganizingFrame

class KeyValue(OrganizingFrame):

    @classmethod
    def required_index_names(cls) -> Sequence[str]:
        return ['key']

    @classmethod
    def required_columns(cls) -> Sequence[str]:
        return ['value']

    @classmethod
    def reserved_columns(cls) -> Sequence[str]:
        return ['note']

# will self-organizing and use 'key' as the index
df = KeyValue.read_csv('example.csv')
print(df.index.names, list(df.columns))  # ['key'], ['value', 'note']
