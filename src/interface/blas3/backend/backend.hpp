/***************************************************************************
 *
 *  @license
 *  Copyright (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCL-BLAS: BLAS implementation using SYCL
 *
 *  @filename backend.hpp
 *
 **************************************************************************/
#if defined(RCAR)
#include "interface/blas3/backend/rcar.hpp"
#elif defined INTEL_GPU
#include "interface/blas3/backend/intel_gpu.hpp"
#elif defined AMD_GPU
#include "interface/blas3/backend/amd_gpu.hpp"
#elif defined ARM_GPU
#include "interface/blas3/backend/arm_gpu.hpp"
#elif defined POWER_VR
#include "interface/blas3/backend/power_vr.hpp"
#elif defined NVIDIA_GPU
#include "interface/blas3/backend/nvidia_gpu.hpp"
#elif defined SYCL_BLAS_FPGA
#include "interface/blas3/backend/fpga.hpp"
#else
#include "interface/blas3/backend/default_cpu.hpp"
#endif
