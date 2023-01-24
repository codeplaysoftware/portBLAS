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
#ifdef RCAR
#include "interface/blas2/backend/rcar.hpp"
#elif INTEL_GPU
#include "interface/blas2/backend/intel_gpu.hpp"
#elif AMD_GPU
#include "interface/blas2/backend/amd_gpu.hpp"
#elif ARM_GPU
#include "interface/blas2/backend/arm_gpu.hpp"
#elif POWER_VR
#include "interface/blas2/backend/power_vr.hpp"
#elif NVIDIA_GPU
#include "interface/blas2/backend/nvidia_gpu.hpp"
#else
#include "interface/blas2/backend/default_cpu.hpp"
#endif
