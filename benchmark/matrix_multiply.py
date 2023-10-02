#!/usr/bin/env python3

import collections
import contextlib
import os
import random
import sys
import time
import urllib
import base64
import pathlib

import scipy as sp
import numpy as np
import scipy.sparse

import sqlalchemy
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy import Column, Integer, Float, create_engine
from sqlalchemy import insert, select
from sqlalchemy.orm import Session, aliased, as_declarative
from sqlalchemy.sql import func
from sqlalchemy.sql.expression import alias

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import ClauseElement, Executable
from sqlalchemy import text as sql_text


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

class Node(Base):
    idnode = Column(Integer(), primary_key=True, nullable=False)
    name = Column(String(36), nullable=False)

class Edge(Base):
    first = Column(Integer(), primary_key=True, nullable=False)
    second = Column(Integer(), primary_key=True, nullable=False)
    value = Column(Float(), nullable=False)

class Result(Base):
    first = Column(Integer(), primary_key=True, nullable=False)
    second = Column(Integer(), primary_key=True, nullable=False)

def sqlite_orm(data, N):
    (ridx, cidx), vals = data
    out = pathlib.Path(os.getenv('PROJECT_DIR')) / "db.sqlite"
    out.unlink(missing_ok=True)
    engine = create_engine("sqlite:///" + str(out), echo=True)

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
                select(A.c.first, B.c.second)
                .select_from(A.join(B, A.c.second == B.c.first))
                .group_by(A.c.first, B.c.second)
            )
            print(stmt, file=sys.stderr)
            with Timer("Matrix Multiply,sqlite orm"):
                result = session.execute(stmt)
                with Timer("RetriveResults,sqlite orm"):
                    result = list(result)
            with Timer("Count"):
                print("Result NNZ", len(result))

    print("")

@contextlib.contextmanager
def postgres_blank_database(unix_socket, tables=None):
    qs = urllib.parse.urlencode(dict(host=unix_socket))
    default_engine = create_engine("postgresql://@/postgres?" + qs, echo=True)
    engine = None
    with default_engine.connect() as conn:
        conn.execute(sql_text("commit"))

        dbname = "tmpdb" + base64.b64encode(os.urandom(12), altchars=b"__").decode(
            "utf-8"
        )

        try:
            conn.execute(sql_text(f'create database "{dbname}"'))

            engine = create_engine(f"postgresql://@/{dbname}?" + qs, echo=True)
            Base.metadata.create_all(engine, tables=tables)

            yield engine

        finally:
            try:
                engine.dispose()
            except:
                pass
            del engine
            conn.execute(sql_text("commit"))
            conn.execute(sql_text(f'drop database if exists "{dbname}"'))


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
                    select(A.c.first, B.c.second)
                    .select_from(A.join(B, A.c.second == B.c.first))
                    .group_by(A.c.first, B.c.second)
                )
                with Timer("Matrix Multiply,postgres orm"):
                    result = session.execute(stmt)
                    with Timer("RetrieveResults,postgres orm"):
                        result = list(result)
                with Timer("Count"):
                    print("Result NNZ", len(result))

        print("")

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
            print(f"{X.shape=}")
            db.save(f"{os.getenv('PROJECT_DIR')}/db.h5")

        with Timer("Matrix Multiply,tensql"):
            qdb = tensql.QIR.QirDatabase(db)
            A = qdb.matrix.aliased("A")
            B = qdb.matrix.aliased("B")

            result = (
                qdb.query(A)
                .join(B, A.cidx == B.ridx)
                .select(A.ridx, B.cidx)
                .group_by(A.ridx, B.cidx)
                .run()
            )
            with Timer("RetrieveResults,tensql"):
                results = list(result)
    print("Result NNZ", len(result))

def tensql_fullmxm(data, N):
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
            print(f"{X.shape=}")
            db.save(f"{os.getenv('PROJECT_DIR')}/db.h5")

        with Timer("Matrix Multiply,tensql"):
            qdb = tensql.QIR.QirDatabase(db)
            A = qdb.matrix.aliased("A")
            B = qdb.matrix.aliased("B")

            result = (
                qdb.query(A)
                .join(B, A.cidx == B.ridx)
                .select(A.ridx, B.cidx, tensql.QIR.Sum(A.value * B.value))
                .group_by(A.ridx, B.cidx)
                .run()
            )
            with Timer("RetrieveResults,tensql"):
                results = list(result)
    print("Result NNZ", len(result))

def tensql_fullmxm_records(data, N):
    (ridx, cidx), vals = data

    import pygraphblas.types
    import tensql
    import tensql.QIR

    with Timer("TenSQL Operations"):
        with Timer("Building sparse matrix"):
            X = pygraphblas.Matrix.from_lists(
                ridx, cidx, vals, nrows=N, ncols=N, typ=pygraphblas.types.FP64
            )

        with Timer("Saving Table,tensql_records"):
            db = tensql.Database()
            print(f"{X.nvals=}")
            t_matrix = db.create_table_from_tensor("matrix", X)
            print(f"{len(t_matrix)=}")
            print(f"{X.shape=}")
            db.save(f"{os.getenv('PROJECT_DIR')}/db.h5")

        with Timer("Matrix Multiply,tensql_records"):
            qdb = tensql.QIR.QirDatabase(db)
            A = qdb.matrix.aliased("A")
            B = qdb.matrix.aliased("B")

            result = (
                qdb.query(A)
                .join(B, A.cidx == B.ridx)
                .select(A.ridx, B.cidx, tensql.QIR.Sum(A.value * B.value))
                .group_by(A.ridx, B.cidx)
                .run()
            )
            with Timer("RetrieveResults,tensql_records"):
                results = result.to_records()
    print("Result NNZ", results.size)


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

        with Timer("Matrix Multiply,pandasql"):
            q = """
      SELECT a.ridx, b.cidx
      FROM matrix AS a
        INNER JOIN matrix AS b ON a.cidx == b.ridx
      GROUP BY a.ridx, b.cidx
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
        tensql,
        postgres_orm,
        sqlite_orm,
        pandasql,
    ]

    benchmark_lookup = {bm.__name__: bm for bm in all_benchmarks}
    run_benchmarks = sys.argv[2:]
    if 'all' in run_benchmarks:
        run_benchmarks = all_benchmarks
    else:
        run_benchmarks = [benchmark_lookup[name] for name in run_benchmarks]

    for benchmark in run_benchmarks:
        benchmark(((ridx, cidx), vals), N)
        print("")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
