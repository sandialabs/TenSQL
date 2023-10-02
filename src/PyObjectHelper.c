#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

void PyObject_EQ(bool* z, int64_t* x, int64_t* y)
{
  PyGILState_STATE gstate;
  gstate = PyGILState_Ensure();

  PyObject* ox = (PyObject*)(x[0]);
  PyObject* oy = (PyObject*)(y[0]);

  Py_INCREF(ox);
  Py_INCREF(oy);
  switch(PyObject_RichCompareBool(ox, oy, Py_EQ))
  {
    case 1:
      z[0] = true;
      break;
    default:
      z[0] = false;
      break;
  }
  Py_DECREF(ox);
  Py_DECREF(oy);

  PyGILState_Release(gstate);
}

void PyObject_NE(bool* z, int64_t* x, int64_t* y)
{
  PyGILState_STATE gstate;
  gstate = PyGILState_Ensure();

  PyObject* ox = (PyObject*)(x[0]);
  PyObject* oy = (PyObject*)(y[0]);

  Py_INCREF(ox);
  Py_INCREF(oy);
  switch(PyObject_RichCompareBool(ox, oy, Py_NE))
  {
    case 1:
      z[0] = true;
      break;
    default:
      z[0] = false;
      break;
  }
  Py_DECREF(ox);
  Py_DECREF(oy);

  PyGILState_Release(gstate);
}

void PyObject_LT(bool* z, int64_t* x, int64_t* y)
{
  PyGILState_STATE gstate;
  gstate = PyGILState_Ensure();

  PyObject* ox = (PyObject*)(x[0]);
  PyObject* oy = (PyObject*)(y[0]);

  Py_INCREF(ox);
  Py_INCREF(oy);
  switch(PyObject_RichCompareBool(ox, oy, Py_LT))
  {
    case 1:
      z[0] = true;
      break;
    default:
      z[0] = false;
      break;
  }
  Py_DECREF(ox);
  Py_DECREF(oy);

  PyGILState_Release(gstate);
}

void PyObject_GT(bool* z, int64_t* x, int64_t* y)
{
  PyGILState_STATE gstate;
  gstate = PyGILState_Ensure();

  PyObject* ox = (PyObject*)(x[0]);
  PyObject* oy = (PyObject*)(y[0]);

  Py_INCREF(ox);
  Py_INCREF(oy);
  switch(PyObject_RichCompareBool(ox, oy, Py_GT))
  {
    case 1:
      z[0] = true;
      break;
    default:
      z[0] = false;
      break;
  }
  Py_DECREF(ox);
  Py_DECREF(oy);

  PyGILState_Release(gstate);
}

void PyObject_GE(bool* z, int64_t* x, int64_t* y)
{
  PyGILState_STATE gstate;
  gstate = PyGILState_Ensure();

  PyObject* ox = (PyObject*)(x[0]);
  PyObject* oy = (PyObject*)(y[0]);

  Py_INCREF(ox);
  Py_INCREF(oy);
  switch(PyObject_RichCompareBool(ox, oy, Py_GE))
  {
    case 1:
      z[0] = true;
      break;
    default:
      z[0] = false;
      break;
  }
  Py_DECREF(ox);
  Py_DECREF(oy);

  PyGILState_Release(gstate);
}

void PyObject_LE(bool* z, int64_t* x, int64_t* y)
{
  PyGILState_STATE gstate;
  gstate = PyGILState_Ensure();

  PyObject* ox = (PyObject*)(x[0]);
  PyObject* oy = (PyObject*)(y[0]);

  Py_INCREF(ox);
  Py_INCREF(oy);
  switch(PyObject_RichCompareBool(ox, oy, Py_LE))
  {
    case 1:
      z[0] = true;
      break;
    default:
      z[0] = false;
      break;
  }
  Py_DECREF(ox);
  Py_DECREF(oy);

  PyGILState_Release(gstate);
}

void PyObject_FIRST(int64_t* z, int64_t* x, int64_t* y)
{
    z[0] = x[0];
}

void PyObject_SECOND(int64_t* z, int64_t* x, int64_t* y)
{
    z[0] = y[0];
}

void PyObject_ANY(int64_t* z, int64_t* x, int64_t* y)
{
    z[0] = x[0];
}

