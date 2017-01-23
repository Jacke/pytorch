#ifndef THP_CUDNN_TYPES_INC
#define THP_CUDNN_TYPES_INC

#include <Python.h>
#include <cstddef>
#include <cudnn.h>
#include "../Types.h"

namespace torch { namespace cudnn {

PyObject * getTensorClass(PyObject *args);
cudnnDataType_t getCudnnDataType(PyObject *tensorClass);
void THVoidTensor_assertContiguous(THVoidTensor *tensor);

}}  // namespace torch::cudnn

#endif
