#include "torch/csrc/utils/pybind.h"
#include "torch/csrc/cuda/comm.h"

#include <chrono>

static const char *_broadcast_docstr =
"Broadcasts a tensor to a number of GPUs.\n"
"Arguments:\n"
"    tensor (Tensor): tensor to broadcast.\n"
"    devices (Iterable): an iterable of devices among which to broadcast.\n"
"      Note that it should be like (src, dst1, dst2, ...), the first element\n"
"      of which is the source device to broadcast from.\n"
"Returns:\n"
"    A tuple containing copies of the ``tensor``, placed on devices\n"
"    corresponding to indices from ``devices``.";


static const char *_broadcast_coalesced_docstr =
"Broadcasts a sequence of tensors to the specified GPUs.\n"
"Small tensors are first coalesced into a buffer to reduce the number\n"
"of synchronizations.\n"
"Arguments:\n"
"    tensors (sequence): tensors to broadcast.\n"
"    devices (Iterable): an iterable of devices among which to broadcast.\n"
"      Note that it should be like (src, dst1, dst2, ...), the first element\n"
"      of which is the source device to broadcast from.\n"
"    buffer_size (int): maximum size of the buffer used for coalescing\n"
"Returns:\n"
"    A tuple containing copies of the ``tensor``, placed on devices\n"
"    corresponding to indices from ``devices``.";

namespace torch { namespace cuda { namespace python {

void initCommMethods(PyObject *module) {
  auto m = py::cast<py::module>(module);
  m.def("_broadcast_coalesced", [](std::vector<at::Tensor>& tensors, std::vector<int64_t> devices, std::size_t buffer_size) {
     return broadcast_coalesced(tensors, devices, buffer_size);
   }, py::arg("tensors"), py::arg("devices"), py::arg("buffer_size") = 10 * 1024 * 1024,
      py::call_guard<py::gil_scoped_release>(), _broadcast_coalesced_docstr)
   .def("_broadcast", [](at::Tensor& tensor, std::vector<int64_t> devices) {
     return broadcast(tensor, devices);
   }, py::call_guard<py::gil_scoped_release>(), _broadcast_docstr);
}

}}}
