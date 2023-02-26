#include "pch.h"
#include <cublas_v2.h>

extern "C" __declspec(dllexport) bool initializeGPUMatrixAcceleration() {
	cublasHandle_t cublasHandle;
	cublasStatus_t status = cublasCreate(&cublasHandle);
	bool initSuccessful = status == CUBLAS_STATUS_SUCCESS;
	cublasDestroy(cublasHandle);
	return initSuccessful;
	//return 100;
}