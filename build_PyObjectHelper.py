import cffi
import pathlib

src_path = pathlib.Path('src/PyObjectHelper.c')
with src_path.open('r') as fin:
  src = fin.read()

ffibuilder = cffi.FFI()
ffibuilder.set_source("tensql._PyObjectHelper", src)
ffibuilder.cdef("""
void PyObject_EQ(bool* z, int64_t* x, int64_t* y);
void PyObject_NE(bool* z, int64_t* x, int64_t* y);
void PyObject_LT(bool* z, int64_t* x, int64_t* y);
void PyObject_GT(bool* z, int64_t* x, int64_t* y);
void PyObject_LE(bool* z, int64_t* x, int64_t* y);
void PyObject_GE(bool* z, int64_t* x, int64_t* y);

void PyObject_ANY(int64_t* z, int64_t* x, int64_t* y);
void PyObject_FIRST(int64_t* z, int64_t* x, int64_t* y);
void PyObject_SECOND(int64_t* z, int64_t* x, int64_t* y);
""")

if __name__ == "__main__":
  ffibuilder.compile()
