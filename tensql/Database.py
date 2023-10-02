import pathlib
from typing import Union

import h5py

from .Table import Table


class Database:
    def __init__(self):
        self.tables = {}

    def create_table(self, name: str, *args):
        assert name not in self.tables
        table = Table(*args)
        self.tables[name] = table
        return table

    def create_table_from_definition(self, name: str, *args, **kwargs) -> Table:
        assert name not in self.tables
        table = Table.from_definition(*args, **kwargs)
        self.tables[name] = table
        return table

    def create_table_from_tensor(self, name: str, *args, **kwargs) -> Table:
        assert name not in self.tables
        table = Table.from_tensor(*args, **kwargs)
        self.tables[name] = table
        return table

    def add_table(self, name: str, table: Table) -> Table:
        self.tables[name] = table
        return table

    def __getitem__(self, name: str) -> Table:
        return self.tables[name]

    def __setitem__(self, name: str, table: Table) -> Table:
        return self.add_table(name, table)

    def __delitem__(self, name: str) -> None:
        del self.tables[name]

    def __contains__(self, name: str) -> bool:
        return name in self.tables

    def save(self, path: Union[pathlib.Path, str], verbose=0):
        if verbose >= 1:
            print("Saving", "Database", str(path))
        with h5py.File(str(path), "w") as fout:
            for table_name, table in self.tables.items():
                if verbose >= 1:
                  print("Saving", "Table", table_name, "with", table.stencil.nvals, "rows")
                table.save_hdf5(fout.create_group(table_name), verbose=verbose-1)

    @classmethod
    def load(cls, path: Union[pathlib.Path, str]):
        db = cls()

        with h5py.File(str(path), "r") as fin:
            for table_name, table_group in fin.items():
                db.tables[table_name] = Table.load_hdf5(table_group)

        return db
