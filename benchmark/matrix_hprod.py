#!/usr/bin/env python3

import collections
import contextlib
import os
import random
import sys
import time
import urllib
import base64
import inspect

import scipy as sp
import numpy as np
import scipy.sparse

import sqlalchemy
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy import Column, Integer, Float, create_engine
from sqlalchemy import insert, select, and_
from sqlalchemy.orm import Session, aliased, as_declarative
from sqlalchemy.sql import func
from sqlalchemy.sql.expression import alias

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import ClauseElement, Executable


class CreateTableAs(Executable, ClauseElement):
    def __init__(self, name, query):
        self.name = name
        self.query = query


@compiles(CreateTableAs)
def _create_table_as(element, compiler, **kw):
    return "CREATE TABLE %s AS %s" % (
        element.name,
        compiler.process(element.query)
    )

class Timer:
    def __init__(self, name):
        self.start_time = None
        self.name = name

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, *args):
        end_time = time.time()
        print(f"{self.name},{end_time - self.start_time},seconds")


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
    first = Column(Integer(), primary_key=True, nullable=False, index=True)
    second = Column(Integer(), primary_key=True, nullable=False, index=True)
    value = Column(Float(), nullable=False)

class Result(Base):
    first = Column(Integer(), primary_key=True, nullable=False)
    second = Column(Integer(), primary_key=True, nullable=False)
    value = Column(Float(), nullable=False)


# if os.path.exists(sys.argv[2]):
#  os.remove(sys.argv[2])


def sqlite_insert_into(data, N):
    (ridx, cidx), vals = data
    engine = create_engine("sqlite:///:memory:", echo=True)

    Base.metadata.create_all(engine)

    with Timer("SQLite Operations"):
        with Session(engine) as session:
            with open(sys.argv[1], "r") as fin:
                with Timer("Reading Data and Building ORM objects"):
                    objects = []
                    for r, c, v in zip(ridx, cidx, vals):
                        objects.append(Edge(first=r, second=c, value=v))
                with Timer("Saving Table,sqlite insert into"):
                    session.bulk_save_objects(objects)
                with Timer("Commit"):
                    session.commit()

        with Session(engine) as session:
            A = alias(Edge, name="A")
            B = alias(Edge, name="B")
            q_stmt = (
                select(A.c.first, A.c.second, A.c.value * B.c.value).select_from(
                    A.join(B, and_(A.c.first == B.c.second, A.c.second == B.c.first))
                )
                #          .group_by(A.c.first, A.c.second)
            )
            stmt = insert(Result).from_select(
                [Result.first, Result.second, Result.value], q_stmt
            )
            print(stmt, file=sys.stderr)
            with Timer("eWiseMult,sqlite insert into"):
                session.execute(stmt)
            with Timer("Commit"):
                session.commit()
            with Timer("Count"):
                print("Result NNZ", session.query(Result).count())

    print("")


def sqlite_orm(data, N):
    (ridx, cidx), vals = data
    engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(engine)

    with Timer("SQLite Operations"):
        with Session(engine) as session:
            with open(sys.argv[1], "r") as fin:
                with Timer("Reading Data and Building ORM objects"):
                    objects = []
                    for r, c, v in zip(ridx, cidx, vals):
                        objects.append(Edge(first=r, second=c, value=v))
                with Timer("Saving Table,sqlite orm"):
                    session.bulk_save_objects(objects)
                with Timer("Commit"):
                    session.commit()

        with Session(engine) as session:
            A = alias(Edge, name="A")
            B = alias(Edge, name="B")
            stmt = (
                select(A.c.first, A.c.second, A.c.value * B.c.value).select_from(
                    A.join(B, and_(A.c.first == B.c.second, A.c.second == B.c.first))
                )
                #          .group_by(A.c.first, A.c.second)
            )
            print(stmt, file=sys.stderr)
            with Timer("eWiseMult,sqlite orm"):
                result = list(session.execute(stmt))
            with Timer("Count"):
                print("Result NNZ", len(result))

    print("")

def sqlite_create_table_as(data, N):
    (ridx, cidx), vals = data
    engine = create_engine("sqlite:///:memory:", echo=True)

    Base.metadata.create_all(engine, tables=[Edge.__table__])

    with Timer("SQLite Operations"):
        with Session(engine) as session:
            with open(sys.argv[1], "r") as fin:
                with Timer("Reading Data and Building ORM objects"):
                    objects = []
                    for r, c, v in zip(ridx, cidx, vals):
                        objects.append(Edge(first=r, second=c, value=v))
                with Timer("Saving Table,sqlite create table as"):
                    session.bulk_save_objects(objects)
                with Timer("Commit"):
                    session.commit()


        with Session(engine) as session:
            A = alias(Edge, name="A")
            B = alias(Edge, name="B")
            stmt = CreateTableAs(
                "Result",
                select(A.c.first, A.c.second, (A.c.value * B.c.value).label("value")).select_from(
                    A.join(B, and_(A.c.first == B.c.second, A.c.second == B.c.first))
                )
            )
            print(stmt, file=sys.stderr)
            with Timer("eWiseMult,sqlite create table as"):
                session.execute(stmt)
            with Timer("Count"):
                print("Result NNZ", session.query(Result).count())

    print("")


@contextlib.contextmanager
def postgres_blank_database(unix_socket, tables=None):
    qs = urllib.parse.urlencode(dict(host=unix_socket))
    default_engine = create_engine("postgresql://@/postgres?" + qs, echo=True)
    engine = None
    with default_engine.connect() as conn:
        conn.execute("commit")

        dbname = "tmpdb" + base64.b64encode(os.urandom(12), altchars=b"__").decode(
            "utf-8"
        )

        try:
            conn.execute(f'create database "{dbname}"')

            engine = create_engine(f"postgresql://@/{dbname}?" + qs)
            Base.metadata.create_all(engine, tables=tables)

            yield engine

        finally:
            try:
                engine.dispose()
            except:
                pass
            del engine
            conn.execute("commit")
            conn.execute(f'drop database if exists "{dbname}"')


def postgres_insert_into(data, N):
    (ridx, cidx), vals = data
    with postgres_blank_database(os.getenv("POSTGRES_SOCKETDIR")) as engine:
        with Timer("Postgres Operations"):
            with Session(engine) as session:
                with open(sys.argv[1], "r") as fin:
                    with Timer("Reading Data and Building ORM objects"):
                        objects = []
                        for r, c, v in zip(ridx, cidx, vals):
                            objects.append(Edge(first=r, second=c, value=v))
                    with Timer("Saving Table,postgres insert into"):
                        session.bulk_save_objects(objects)
                    with Timer("Commit"):
                        session.commit()

            with Session(engine) as session:
                A = alias(Edge, name="A")
                B = alias(Edge, name="B")
                q_stmt = (
                    select(A.c.first, A.c.second, A.c.value * B.c.value).select_from(
                        A.join(
                            B, and_(A.c.first == B.c.second, A.c.second == B.c.first)
                        )
                    )
                    #            .group_by(A.c.first, A.c.second)
                )
                stmt = insert(Result).from_select(
                    [Result.first, Result.second, Result.value], q_stmt
                )
                print(stmt, file=sys.stderr)
                with Timer("eWiseMult,postgres insert into"):
                    session.execute(stmt)
                with Timer("Commit"):
                    session.commit()
                with Timer("Count"):
                    print("Result NNZ", session.query(Result).count())

        print("")


def postgres_orm(data, N):
    (ridx, cidx), vals = data
    with postgres_blank_database(os.getenv("POSTGRES_SOCKETDIR")) as engine:
        with Timer("Postgres Operations"):
            with Session(engine) as session:
                with open(sys.argv[1], "r") as fin:
                    with Timer("Reading Data and Building ORM objects"):
                        objects = []
                        for r, c, v in zip(ridx, cidx, vals):
                            objects.append(Edge(first=r, second=c, value=v))
                    with Timer("Saving Table,postgres orm"):
                        session.bulk_save_objects(objects)
                    with Timer("Commit"):
                        session.commit()

            with Session(engine) as session:
                A = alias(Edge, name="A")
                B = alias(Edge, name="B")
                stmt = (
                    select(A.c.first, A.c.second, (A.c.value * B.c.value).label("value")).select_from(
                        A.join(
                            B, and_(A.c.first == B.c.second, A.c.second == B.c.first)
                        )
                    )
                    #            .group_by(A.c.first, A.c.second)
                )
                print(stmt, file=sys.stderr)
                with Timer("eWiseMult,postgres orm"):
                    result = list(session.execute(stmt))
                with Timer("Count"):
                    print("Result NNZ", len(result))

        print("")

def postgres_create_table_as(data, N):
    (ridx, cidx), vals = data
    with postgres_blank_database(os.getenv("POSTGRES_SOCKETDIR"), tables=[Edge.__table__]) as engine:
        with Timer("Postgres Operations"):
            with Session(engine) as session:
                with open(sys.argv[1], "r") as fin:
                    with Timer("Reading Data and Building ORM objects"):
                        objects = []
                        for r, c, v in zip(ridx, cidx, vals):
                            objects.append(Edge(first=r, second=c, value=v))
                    with Timer("Saving Table,postgres create table as"):
                        session.bulk_save_objects(objects)
                    with Timer("Commit"):
                        session.commit()

            with Session(engine) as session:
                A = alias(Edge, name="A")
                B = alias(Edge, name="B")
                stmt = CreateTableAs(
                    "Result",
                    select(A.c.first, A.c.second, (A.c.value * B.c.value).label("value")).select_from(
                        A.join(
                            B, and_(A.c.first == B.c.second, A.c.second == B.c.first)
                        )
                    )
                )
                print(stmt, file=sys.stderr)
                with Timer("eWiseMult,postgres create table as"):
                    session.execute(stmt)
                with Timer("Count"):
                    print("Result NNZ", session.query(Result).count())

        print("")


def scipy_csr(data, N):
    (ridx, cidx), vals = data
    with Timer("Scipy CSR Operations"):
        with Timer("Building sparse matrix"):
            X = sp.sparse.csr_matrix(
                (vals, (ridx, cidx)), dtype=np.float64, shape=(N, N)
            )
        print(len(ridx), len(cidx), len(vals), N)

        with Timer("eWiseMult,scipy csr"):
            ret = X.multiply(X.T)
    print("Result NNZ", ret.nnz)


def scipy_coo(data, N):
    (ridx, cidx), vals = data
    with Timer("Scipy COO Operations"):
        with Timer("Building sparse matrix"):
            X = sp.sparse.coo_matrix(
                (vals, (ridx, cidx)), dtype=np.float64, shape=(N, N)
            )
        print(len(ridx), len(cidx), len(vals), N)

        with Timer("eWiseMult,scipy coo"):
            ret = X.multiply(X.T)
    print("Result NNZ", ret.nnz)


def pygraphblas(data, N):
    (ridx, cidx), vals = data
    import pygraphblas

    with Timer("PyGraphBLAS Operations"):
        with Timer("Building sparse matrix"):
            X = pygraphblas.Matrix.from_lists(
                ridx, cidx, vals, nrows=N, ncols=N, typ=pygraphblas.types.FP64
            )
            X = X.dup().cast(pygraphblas.types.FP64)

        with Timer("eWiseMult,pygraphblas"):
            ret = X.emult(X.T)
    print("Result NNZ", ret.nvals)

def tensql_create_table_as(data, N):
    (ridx, cidx), vals = data

    import pygraphblas.types
    import tensql
    import tensql.QIR

    with Timer("TenSQL Operations"):
        with Timer("Building sparse matrix"):
            X = pygraphblas.Matrix.from_lists(
                ridx, cidx, vals, nrows=N, ncols=N, typ=pygraphblas.types.FP64
            )

        with Timer("Saving Table,tensql_create_table_as"):
            db = tensql.Database()
            print(f"{X.nvals=}")
            t_matrix = db.create_table_from_tensor("matrix", X)
            print(f"{len(t_matrix)=}")
            db.save(f"{os.getenv('PROJECT_DIR')}/db.h5")

        with Timer("eWiseMult,tensql_create_table_as"):
            qdb = tensql.QIR.QirDatabase(db)
            A = qdb.matrix.aliased("A")
            B = qdb.matrix.aliased("B")

            result = (
                qdb.query(A)
                .join(
                    B,
                    tensql.QIR.LogicalAnd(
                        A.ridx == B.cidx, A.cidx == B.ridx, name="value"
                    ),
                )
                .select(A.ridx, A.cidx, A.value * B.value)
                .run()
            )

            db.save(f"{os.getenv('PROJECT_DIR')}/db.h5")
    print("Result NNZ", result.count())

def tensql(data, N):
    (ridx, cidx), vals = data

    import pygraphblas.types
    import tensql
    import tensql.QIR

    with Timer("TenSQL Operations"):
        with Timer("Building sparse matrix"):
            X = pygraphblas.Matrix.from_lists(
                ridx, cidx, vals, nrows=N, ncols=N, typ=pygraphblas.types.FP64
            )

        with Timer("Saving Table,tensql"):
            db = tensql.Database()
            print(f"{X.nvals=}")
            t_matrix = db.create_table_from_tensor("matrix", X)
            print(f"{len(t_matrix)=}")
            db.save(f"{os.getenv('PROJECT_DIR')}/db.h5")

        with Timer("eWiseMult,tensql"):
            qdb = tensql.QIR.QirDatabase(db)
            A = qdb.matrix.aliased("A")
            B = qdb.matrix.aliased("B")

            result = (
                qdb.query(A)
                .join(
                    B,
                    tensql.QIR.LogicalAnd(
                        A.ridx == B.cidx, A.cidx == B.ridx, name="value"
                    ),
                )
                .select(A.ridx, A.cidx, A.value * B.value)
                .run()
            )

    print("Result NNZ", result.count())


def pandasql(data, N):
    (ridx, cidx), vals = data
    import pandasql
    import pandas as pd

    with Timer("PandaSQL Operations"):
        with Timer("Building dataframe"):
            matrix = pd.DataFrame(
                zip(ridx, cidx, vals),
                columns=["ridx", "cidx", "value"],
            )

        with Timer("eWiseMult,pandasql"):
            q = """
      SELECT a.ridx, a.cidx, a.value * b.value AS value
      FROM matrix AS a
        INNER JOIN matrix AS b ON a.ridx == b.cidx AND a.cidx == b.ridx
      """

            result: pd.DataFrame = pandasql.sqldf(q)

    print("Result NNZ", len(result))


def main():
    with Timer("Reading data"):
        maxidx = 0
        ridx, cidx, vals = [], [], []
        condenser = Condenser()
        is_unique = UniqueChecker()
        with open(sys.argv[1], "r") as fin:
            for line in fin:
                first, second = tuple(condenser[int(x)] for x in line.strip().split())
                if is_unique(first, second):
                    ridx.append(first)
                    cidx.append(second)
                    vals.append(random.random())
                    maxidx = max(maxidx, first, second)
        N = maxidx + 1
    print("")

    all_benchmarks = [
        pygraphblas,
        scipy_coo,
        scipy_csr,
        tensql,
        postgres_orm,
        postgres_insert_into,
        postgres_create_table_as,
        sqlite_orm,
        sqlite_insert_into,
        sqlite_create_table_as,
        pandasql,
    ]

    benchmark_lookup = {bm.__name__: bm for bm in all_benchmarks}
    run_benchmarks = set(sys.argv[2:])
    if 'all' in run_benchmarks:
        run_benchmarks = set(all_benchmarks)
    else:
        run_benchmarks = set(benchmark_lookup[name] for name in run_benchmarks)

    for benchmark in run_benchmarks:
        benchmark(((ridx, cidx), vals), N)
        print("")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
