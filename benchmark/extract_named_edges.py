#!/usr/bin/env python3

import base64
import collections
import contextlib
import inspect
import os
import random
import sys
import time
import urllib
import uuid
import pathlib

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

class Node(Base):
    idnode = Column(Integer(), primary_key=True, nullable=False)
    guid = Column(String(36), nullable=False)

class Edge(Base):
    first = Column(Integer(), primary_key=True, nullable=False)
    second = Column(Integer(), primary_key=True, nullable=False)
    value = Column(Float(), nullable=False)


def sqlite_orm(data, uuids):
    (ridx, cidx), vals = data
    N = len(uuids)
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
                    for k,v in uuids.items():
                        objects.append(Node(idnode=k, guid=v))
                with Timer("Saving Table,sqlite orm"):
                    session.bulk_save_objects(objects)
                with Timer("Commit"):
                    session.commit()

        with Session(engine) as session:
            A = alias(Edge, name="A")
            x = alias(Node, name="x")
            y = alias(Node, name="y")

            stmt = (
                select(x.c.guid.label('first'), y.c.guid.label('second')).select_from(
                    A.join(x, A.c.first == x.c.idnode).join(y, A.c.second == y.c.idnode)
                )
            )
            print(stmt, file=sys.stderr)
            with Timer("NamedEdges,sqlite orm"):
                result = session.execute(stmt)
                with Timer("RetrieveResults,sqlite orm"):
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
        conn.execute("commit")

        dbname = "tmpdb" + base64.b64encode(os.urandom(12), altchars=b"__").decode(
            "utf-8"
        )

        try:
            conn.execute(f'create database "{dbname}"')

            engine = create_engine(f"postgresql://@/{dbname}?" + qs, echo=True)
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

def postgres_orm(data, uuids):
    (ridx, cidx), vals = data
    N = len(uuids)
    with postgres_blank_database(os.getenv("POSTGRES_SOCKETDIR")) as engine:
        with Timer("Postgres Operations"):
            with Session(engine) as session:
                with open(sys.argv[1], "r") as fin:
                    with Timer("Reading Data and Building ORM objects"):
                        objects = []
                        for r, c, v in zip(ridx, cidx, vals):
                            objects.append(Edge(first=r, second=c, value=v))
                        for k,v in uuids.items():
                            objects.append(Node(idnode=k, guid=v))
                    with Timer("Saving Table,postgres orm"):
                        session.bulk_save_objects(objects)
                    with Timer("Commit"):
                        session.commit()

            with Session(engine) as session:
                A = alias(Edge, name="A")
                x = alias(Node, name="x")
                y = alias(Node, name="y")

                stmt = (
                    select(x.c.guid.label('first'), y.c.guid.label('second')).select_from(
                        A.join(x, A.c.first == x.c.idnode).join(y, A.c.second == y.c.idnode)
                    )
                )
                print(stmt, file=sys.stderr)
                with Timer("NamedEdges,postgres orm"):
                    result = session.execute(stmt)
                    with Timer("RetrieveResults,postgres orm"):
                        result = list(result)
                with Timer("Count"):
                    print("Result NNZ", len(result))

        print("")

def tensql(data, uuids):
    (ridx, cidx), vals = data
    N = len(uuids)

    import pygraphblas.types
    import tensql
    import tensql.QIR

    tString = tensql.Types.Text()

    with Timer("TenSQL Operations"):
        with Timer("Building pygraphblas data"):
            A = pygraphblas.Matrix.from_lists(
                ridx, cidx, vals, nrows=N, ncols=N, typ=pygraphblas.types.FP64
            )

            uuid_objs = tuple({'idx': key, 'value': value} for key, value in uuids.items())

        with Timer("Saving Table,tensql"):
            db = tensql.Database()
            qdb = tensql.QIR.QirDatabase(db)
            t_edge = db.create_table_from_tensor("edge", A)
            t_node = db.create_table_from_definition(
                "node",
                {
                    'idx': tensql.Types.BigInt(),
                    'value': tensql.Types.Text()
                },
                ['idx'],
                shape = [N]
            )
            qdb.insert(qdb.node).values(uuid_objs).run()

            print(f"{len(t_edge)=}, {len(t_node)=}")
            db.save(f"{os.getenv('PROJECT_DIR')}/db.h5")

        with Timer("NamedEdges,tensql"):
            A = qdb.edge.aliased("A")
            x = qdb.node.aliased("x")
            y = qdb.node.aliased("y")

            result = (
                qdb.query(A)
                .join(x, x.idx == A.ridx)
                .join(y, y.idx == A.cidx)
                .select(A.ridx, A.cidx, x.value.alias('first'), y.value.alias('second'))
                .run()
            )

            with Timer("RetrieveResults,tensql"):
                results = [x for x in result]

    print("Result NNZ", result.count())

def tensql_records(data, uuids):
    (ridx, cidx), vals = data
    N = len(uuids)

    import pygraphblas.types
    import tensql
    import tensql.QIR

    tString = tensql.Types.Text()

    with Timer("TenSQL Operations"):
        with Timer("Building pygraphblas data"):
            A = pygraphblas.Matrix.from_lists(
                ridx, cidx, vals, nrows=N, ncols=N, typ=pygraphblas.types.FP64
            )

            uuid_objs = tuple({'idx': key, 'value': value} for key, value in uuids.items())

        with Timer("Saving Table,tensql_records"):
            db = tensql.Database()
            qdb = tensql.QIR.QirDatabase(db)
            t_edge = db.create_table_from_tensor("edge", A)
            t_node = db.create_table_from_definition(
                "node",
                {
                    'idx': tensql.Types.BigInt(),
                    'value': tensql.Types.Text()
                },
                ['idx'],
                shape = [N]
            )
            qdb.insert(qdb.node).values(uuid_objs).run()

            print(f"{len(t_edge)=}, {len(t_node)=}")
            db.save(f"{os.getenv('PROJECT_DIR')}/db.h5")

        with Timer("NamedEdges,tensql_records"):
            A = qdb.edge.aliased("A")
            x = qdb.node.aliased("x")
            y = qdb.node.aliased("y")

            result = (
                qdb.query(A)
                .join(x, x.idx == A.ridx)
                .join(y, y.idx == A.cidx)
                .select(A.ridx, A.cidx, x.value.alias('first'), y.value.alias('second'))
                .run()
            )

            with Timer("RetrieveResults,tensql_records"):
                results = result.to_records()

    print("Result NNZ", results.size)


def pandasql(data, uuids):
    (ridx, cidx), vals = data
    N = len(uuids)
    import pandasql
    import pandas as pd

    with Timer("PandaSQL Operations"):
        with Timer("Building dataframe"):
            A = pd.DataFrame(
                zip(ridx, cidx, vals),
                columns=["ridx", "cidx", "value"],
            )
            x = pd.DataFrame(
                uuids.items(),
                columns=["idx", "name"]
            )
            y = pd.DataFrame(
                uuids.items(),
                columns=["idx", "name"]
            )

        with Timer("NamedEdges,pandasql"):
            q = """
      SELECT x.name, y.name
      FROM A
        INNER JOIN x ON A.ridx == x.idx
        INNER JOIN y ON A.cidx == y.idx
      """

            result: pd.DataFrame = pandasql.sqldf(q)

    print("Result NNZ", len(result))


def main():
    with Timer("Reading data"):
        maxidx = 0
        ridx, cidx, vals = [], [], []
        condenser = Condenser()
        edge_unique = UniqueChecker()
        node_unique = UniqueChecker()
        uuids = {}
        with open(sys.argv[1], "r") as fin:
            for line in fin:
                first, second = tuple(condenser[int(x)] for x in line.strip().split())
                if node_unique(first):
                    uuids[first] = str(uuid.uuid4())
                if node_unique(second):
                    uuids[second] = str(uuid.uuid4())
                if edge_unique(first, second):
                    ridx.append(first)
                    cidx.append(second)
                    vals.append(random.random())
                    maxidx = max(maxidx, first, second)
        N = len(uuids)
    print("")

    all_benchmarks = [
        tensql,
        postgres_orm,
        sqlite_orm,
        pandasql,
    ]

    benchmark_lookup = {bm.__name__: bm for bm in all_benchmarks}
    run_benchmarks = set(sys.argv[2:])
    if 'all' in run_benchmarks:
        run_benchmarks = set(all_benchmarks)
    else:
        run_benchmarks = set(benchmark_lookup[name] for name in run_benchmarks)

    for benchmark in run_benchmarks:
        benchmark(((ridx, cidx), vals), uuids)
        print("")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
