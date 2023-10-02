#!/usr/bin/env python3

import os
import random
import sys
import time
import collections

import sqlalchemy
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy import Column, Integer, Float, create_engine
from sqlalchemy import insert, select
from sqlalchemy.orm import Session, aliased, as_declarative
from sqlalchemy.sql import func
from sqlalchemy.sql.expression import alias, and_

import numpy as np
import scipy as sp
import scipy.sparse


class Timer:
    def __init__(self, name):
        self.start_time = None
        self.name = name

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, *args):
        end_time = time.time()
        print(f"{self.name} ran for {end_time - self.start_time} seconds")


class Counter:
    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count = (ret := self.count) + 1
        return ret


class Condenser(collections.defaultdict):
    def __init__(self):
        super().__init__(Counter())


class UniqueChecker:
    def __init__(self):
        self.data = {}

    def __call__(self, *args):
        val = random.randint(0, 2**63 - 1)
        return self.data.setdefault(args, val) == val


@as_declarative()
class Base(object):
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()


class Edge(Base):
    first = Column(Integer(), primary_key=True, nullable=False)
    second = Column(Integer(), primary_key=True, nullable=False)
    value = Column(Float(), nullable=False)


class Result(Base):
    first = Column(Integer(), primary_key=True, nullable=False)
    second = Column(Integer(), primary_key=True, nullable=False)
    value = Column(Float(), nullable=False)


# if os.path.exists(sys.argv[2]):
#  os.remove(sys.argv[2])


def sqlite():
    engine = create_engine("sqlite://")

    Base.metadata.create_all(engine)

    with Timer("SQLite Operations"):
        with Session(engine) as session:
            with open(sys.argv[1], "r") as fin:
                with Timer("Reading Data and Building ORM objects"):
                    condenser = Condenser()
                    is_unique = UniqueChecker()
                    objects = []
                    for line in fin:
                        first, second = [
                            condenser[int(x)] for x in line.strip().split()
                        ]
                        if is_unique(first, second):
                            objects.append(
                                Edge(first=first, second=second, value=random.random())
                            )
                with Timer("Bulk Save Objects"):
                    session.bulk_save_objects(objects)
                with Timer("Commit"):
                    session.commit()

        with Session(engine) as session:
            A = alias(Edge, name="A")
            B = alias(Edge, name="B")
            stmt = insert(Result).from_select(
                [Result.first, Result.second, Result.value],
                select(A.c.first, A.c.second, A.c.value + B.c.value)
                .select_from(
                    A.join(B, and_(A.c.first == B.c.second, A.c.second == B.c.first))
                )
                .group_by(A.c.first, A.c.second),
            )
            print(stmt, file=sys.stderr)
            with Timer("Matrix eWiseAdd"):
                result = session.execute(stmt)
            session.commit()

    with Session(engine) as session:
        print(
            "Result NNZ",
            session.execute(select(func.count()).select_from(Result)).scalar(),
        )


def scipy_csr():
    with Timer("Scipy CSR Operations"):
        with Timer("Reading data"):
            ridx, cidx, vals = [], [], []
            maxidx = 0
            condenser = Condenser()
            is_unique = UniqueChecker()
            with open(sys.argv[1], "r") as fin:
                for line in fin:
                    first, second = tuple(
                        condenser[int(x)] for x in line.strip().split()
                    )
                    if is_unique(first, second):
                        ridx.append(first)
                        cidx.append(second)
                        vals.append(random.random())
                        maxidx = max(maxidx, first, second)

        N = maxidx + 1

        with Timer("Building sparse matrix"):
            X = sp.sparse.csr_matrix(
                (vals, (ridx, cidx)), dtype=np.float64, shape=(N, N)
            )
        print(len(ridx), len(cidx), len(vals), N)

        with Timer("Matrix eWiseAdd"):
            ret = X + X.T
    print("Result NNZ", ret.nnz)


def scipy_coo():
    with Timer("Scipy COO Operations"):
        with Timer("Reading data"):
            ridx, cidx, vals = [], [], []
            maxidx = 0
            condenser = Condenser()
            is_unique = UniqueChecker()
            with open(sys.argv[1], "r") as fin:
                for line in fin:
                    first, second = tuple(
                        condenser[int(x)] for x in line.strip().split()
                    )
                    if is_unique(first, second):
                        ridx.append(first)
                        cidx.append(second)
                        vals.append(random.random())
                        maxidx = max(maxidx, first, second)

        N = maxidx + 1

        with Timer("Building sparse matrix"):
            X = sp.sparse.coo_matrix(
                (vals, (ridx, cidx)), dtype=np.float64, shape=(N, N)
            )
        print(len(ridx), len(cidx), len(vals), N)

        with Timer("Matrix eWiseAdd"):
            ret = X + X.T
    print("Result NNZ", ret.nnz)


def pygraphblas():
    import pygraphblas

    with Timer("PyGraphBLAS Operations"):
        with Timer("Reading data"):
            maxidx = 0
            ridx, cidx, vals = [], [], []
            condenser = Condenser()
            is_unique = UniqueChecker()
            with open(sys.argv[1], "r") as fin:
                for line in fin:
                    first, second = tuple(
                        condenser[int(x)] for x in line.strip().split()
                    )
                    if is_unique(first, second):
                        ridx.append(first)
                        cidx.append(second)
                        vals.append(random.random())
                        maxidx = max(maxidx, first, second)

        N = maxidx + 1

        with Timer("Building sparse matrix"):
            X = pygraphblas.Matrix.from_lists(
                ridx, cidx, vals, nrows=N, ncols=N, typ=pygraphblas.types.FP64
            )

        with Timer("Matrix eWiseAdd"):
            ret = X.eadd(X.T)
    print("Result NNZ", ret.nvals)


# TODO
# def tensql():
# import pygraphblas.types
# import tensql
# import tensql.QIR
# with Timer("TenSQL Operations"):
#   with Timer("Reading data"):
#     maxidx = 0
#     ridx, cidx, vals = [], [], []
#     condenser = Condenser()
#     is_unique = UniqueChecker()
#     with open(sys.argv[1], 'r') as fin:
#       for line in fin:
#         first, second = tuple(condenser[int(x)] for x in line.strip().split())
#         if is_unique(first, second):
#           ridx.append(first)
#           cidx.append(second)
#           vals.append(random.random())
#           maxidx = max(maxidx, first, second)

#   with Timer("Building sparse matrix"):
#     N = maxidx + 1
#     X = pygraphblas.Matrix.from_lists(ridx, cidx, vals, nrows=N, ncols=N, typ=pygraphblas.types.FP64)

#   with Timer("Building Table"):
#     db = tensql.Database()
#     print(f"{X.nvals=}")
#     t_matrix = db.create_table_from_tensor("matrix", X)
#     print(f"{len(t_matrix)=}")

#   with Timer("Matrix eWiseAdd"):
#     qdb = tensql.QIR.QirDatabase(db)
#     A = qdb.matrix.aliased("A")
#     B = qdb.matrix.aliased("B")

#     result = (
#       qdb.query(A)
#         .join(B, tensql.QIR.LogicalAnd(A.ridx == B.cidx, A.cidx == B.ridx, name='value'))
#         .select(A.ridx, A.cidx, A.value + B.value)
#         .run()
#     )
# print("Result NNZ", result.count())


def main():
    for run_benchmark in [pygraphblas, scipy_coo, scipy_csr, sqlite]:
        run_benchmark()
        print("")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
