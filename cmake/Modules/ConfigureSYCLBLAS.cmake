
# We add some flags to workaround OpenCL platform bugs, see ComputeCpp documentation
# COMPUTECPP_USER_FLAGS are used when calling add_sycl_to_target
set(COMPUTECPP_USER_FLAGS -no-serial-memop -Xclang -cl-mad-enable -O3)

# Check to see if we've disabled double support in the tests
option(NO_DOUBLE_SUPPORT "Disable double support when testing." off)
if(NO_DOUBLE_SUPPORT)
  # Define NO_DOUBLE_SUPPORT for the host cxx compiler
  add_definitions(-DNO_DOUBLE_SUPPORT)
endif()

# If the user has specified a specific workgroup size for tests, pass that on to the compiler
if(WG_SIZE)
  add_definitions(-DWG_SIZE=${WG_SIZE})
endif()

# If the user has specified that we should use naive gemm, enable that
option(NAIVE_GEMM "Default to naive GEMM implementations" off)
if(NAIVE_GEMM)
  add_definitions(-DNAIVE_GEMM)
endif()

if(DEFINED TARGET)
message(STATUS "TARGET is defined")
  if(${TARGET} STREQUAL "INTEL_GPU")
    message(STATUS "${TARGET} device is chosen")
    add_definitions(-DINTEL_GPU)
  # If the user has specified  RCAR as a target backend the optimisation for all other device will be disabled
  elseif(${TARGET} STREQUAL "RCAR")
    message(STATUS "${TARGET} device is chosen")
    add_definitions(-DRCAR)
  else()
    message(STATUS "No specific TARGET is defined. TARGET will be selected at runtime.")
    add_definitions(-DDYNAMIC)
  endif()
 else()
  message(STATUS "No specific TARGET is defined. TARGET will be selected at runtime.")
  add_definitions(-DDYNAMIC)
endif()
