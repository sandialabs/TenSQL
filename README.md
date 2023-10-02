# TenSQL
Relational Database Management Systems (RDBMS) have been the most prominent
form of database in the world for several decades. While relational databases
are often applied within high-frequency/low-volume transactional applications
such as website backends, the poor performance of relational databases on
low-frequency/high-volume queries often precludes their application to big data
analysis fields like graph analytics. This work explores the construction of an
RDBMS solution that uses the GraphBLAS API to execute Structured Query Language
(SQL) in an effort to improve performance on high-volume queries. Tables are
redefined to be collections of sparse scalars, vectors, matrices, and more
generally sparse tensors. The explicit values (nonzeros) in these sparse
tensors define the rows and NULL values within the tables. A prototype database
called TenSQL was constructed and evaluated against several SQL implementations
including PostgreSQL. Preliminary results comparing the performance on queries
common in graph analysis applications offer performance improvements as high as
1,400x over PostgreSQL for moderately sized datasets when returning results in
a columnar format.

## Authors
TenSQL was created by Sandia National Laboratories, with assistance provided by
the University of Utah.  

## Installation
TenSQL has only been tested with Python 3.9.  Python 3.10 is too new for the
version of numpy supported by pygraphblas.  

To install from PyPI:
```
pip install tensql
```

To install from source:
```
git clone 'https://github.com/sandialabs/TenSQL.git'
cd TenSQL
pip install .
```

## Testing
To run the tests, you must first clone the sourcecode from github, and then
build the extensions and install testing dependencies.
```
git clone 'https://github.com/sandialabs/TenSQL.git'
cd TenSQL
python setup.py build_ext --inplace
pip install -e ".[test]"
```

The tests can then be run either with the `run_tests.py` script which outputs
code coverage information:
```
python3 run_tests.py
```

Or via python's built-in unittest module
```
python3 -m unittest -v tensql.test
```

Specific tests can be run via the unittest module:
```
python3 -m unittest -v tensql.test.test_queries.xAy.TestQuery_xAy
```

Note: Certain tests for memory leaks can take about a minute to execute.

## Running Benchamrks
To run the benchmarks, you must first clone the sourcecode from github, and then
build the extensions and install testing dependencies.
```
git clone 'https://github.com/sandialabs/TenSQL.git'
cd TenSQL
python setup.py build_ext --inplace
pip install -e ".[test,benchmark]"
```

You must also install PostgreSQL 15 to run the postgres tests.

Once installed, you can run the benchmarks via slurm with:
```
bash download_benchmark_data.sh
bash benchmark_twohop.sh
bash benchmark_ingest_and_named_edges.sh
```

Alternatively, you can run single benchmarks (without slurm) like this:
```
bash download_benchmark_data.sh
bash single_twohop.sh "`pwd`/tmp" "`pwd`/results" all
bash single_ingest_and_named_edges.sh "`pwd`/tmp" "`pwd`/results" all
```

Note: You will likely need to tune the settings in `postgresql.conf` if your
system has less memory than our benchmarking system.

## Citing TenSQL
TenSQL was described in the paper "An SQL Database Built on GraphBLAS", which
was accepeted by the IEEE High Performance Extreme Computing Virtual Conference
in September 2023.  It has not yet been published in IEEE Xplore.
```
Roose, J. P., Vaidya, M., Sadayappan, P., & Rajamanickam, S. (2023). TenSQL: An SQL Database Built on GraphBLAS. 
IEEE High Performance Extreme Computing Virtual Conference, forthcoming.
```
