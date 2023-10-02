#ifndef PYTHON_OBJ
#define PYTHON_OBJ

#include <stddef.h> //For size_t
#include <stdint.h> //For intptr_t
#include <Python.h>
#include <iostream>

class PythonObj
{
    public:
        PyObject* object;

        PythonObj() : 
            object(NULL)
        {
        }

        PythonObj(const PythonObj &copy) :
            object(NULL)
        {
            set(copy.object);
        }

        PythonObj(PyObject* o) :
            object(NULL)
        {
            set(o);
        }

        void set(PyObject* o) {
            if(object != NULL)
            {
                //std::cout << "DECREF " << this << " " << object << std::endl;
                Py_DECREF(object);
            }

            object = o;

            if(object != NULL)
            {
                //std::cout << "INCREF " << this << " " << object << std::endl;
                Py_INCREF(object);
            }

        }

        PyObject* toPython(bool incref=false) const {
            if(incref)
              Py_INCREF(object);
            return object;
        }

        ssize_t getRefCount() {
            return Py_REFCNT(object);
        }

        intptr_t toInteger() const {
            return (intptr_t) object;
        }

        ~PythonObj() {
            //std::cout << "Deallocating PythonObj " << this << " " << object << std::endl;
            set(NULL);
        }
};

bool operator==(const PythonObj &lhs, const PythonObj &rhs) {
    return lhs.object == rhs.object;
}

bool operator!=(const PythonObj &lhs, const PythonObj &rhs) {
    return lhs.object != rhs.object;
}

bool operator<(const PythonObj &lhs, const PythonObj &rhs) {
    return lhs.object < rhs.object;
}

bool operator>(const PythonObj &lhs, const PythonObj &rhs) {
    return lhs.object > rhs.object;
}

bool operator<=(const PythonObj &lhs, const PythonObj &rhs) {
    return lhs.object <= rhs.object;
}

bool operator>=(const PythonObj &lhs, const PythonObj &rhs) {
    return lhs.object >= rhs.object;
}

namespace std {
template <>
struct hash<PythonObj> {
    auto operator()(const PythonObj &obj) -> size_t {
        return obj.toInteger();
    }
};
}

#endif
