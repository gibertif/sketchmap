#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "dimproj.hpp"
void mod_dimproj(py::module &);

PYBIND11_MODULE(sketchmap,pymod) {
	/*
	py::module pymod("sketchmap",R"sketchmap(
Sketchmap
-------------------
General package containing Sketchmap tools.
        )sketchmap");
	*/	
	mod_dimproj(pymod);
}
