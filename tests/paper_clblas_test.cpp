#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <CL/sycl.hpp>
#include <clBLAS.h>

using namespace cl::sycl;
using namespace blas;

#define DEF_NUM_ELEM 1200
#define DEF_STRIDE 1
#define ERROR_ALLOWED 1.0E-6
// #define SHOW_VALUES   1

#define SHOW_TIMES 1  // If it exists, the code prints the execution time
// The ... should be changed by the corresponding routine
#define NUMBER_REPEATS 2  // Number of times the computations are made
// If it is greater than 1, the compile time is not considered

// #########################

int main(int argc, char *argv[]) {
  size_t numE, strd, sizeV, returnVal = 0;
  if (argc == 1) {
    numE = DEF_NUM_ELEM;
    strd = DEF_STRIDE;
  } else if (argc == 2) {
    numE = atoi(argv[1]);
    strd = DEF_STRIDE;
  } else if (argc == 3) {
    numE = atoi(argv[1]);
    strd = atoi(argv[2]);
  } else {
    std::cout << "ERROR!! --> Incorrect number of input parameters"
              << std::endl;
    returnVal = 1;
  }
  if (returnVal == 0) {
    sizeV = numE * strd;
#ifdef SHOW_TIMES
    // VARIABLES FOR TIMING
    std::chrono::time_point<std::chrono::steady_clock> t_start, t_stop;
    std::chrono::duration<double> t0_copy, t0_axpy, t0_add;
    std::chrono::duration<double> t1_copy, t1_axpy, t1_add;
    std::chrono::duration<double> t2_copy, t2_axpy, t2_add;
    std::chrono::duration<double> t3_copy, t3_axpy, t3_add;
#endif

    // CREATING DATA
    std::vector<double> vX1(sizeV);
    std::vector<double> vY1(sizeV);
    std::vector<double> vZ1(sizeV);
    std::vector<double> vS1(sizeV);
    std::vector<double> vX2(sizeV);
    std::vector<double> vY2(sizeV);
    std::vector<double> vZ2(sizeV);
    std::vector<double> vS2(sizeV);
    std::vector<double> vX3(sizeV);
    std::vector<double> vY3(sizeV);
    std::vector<double> vZ3(sizeV);
    std::vector<double> vS3(sizeV);
    std::vector<double> vX4(sizeV);
    std::vector<double> vY4(sizeV);
    std::vector<double> vZ4(sizeV);
    std::vector<double> vS4(sizeV);

    // INITIALIZING DATA
    size_t vSeed, gap;
    double minV, maxV;

    vSeed = 1;
    minV = -10.0;
    maxV = 10.0;
    gap = (size_t)(maxV - minV + 1);
    srand(vSeed);
    std::for_each(std::begin(vX1), std::end(vX1),
                  [&](double &elem) { elem = minV + (double)(rand() % gap); });
    std::for_each(std::begin(vX2), std::end(vX2),
                  [&](double &elem) { elem = minV + (double)(rand() % gap); });
    std::for_each(std::begin(vX3), std::end(vX3),
                  [&](double &elem) { elem = minV + (double)(rand() % gap); });
    std::for_each(std::begin(vX4), std::end(vX4),
                  [&](double &elem) { elem = minV + (double)(rand() % gap); });

    vSeed = 1;
    minV = -30.0;
    maxV = 10.0;
    gap = (size_t)(maxV - minV + 1);
    srand(vSeed);
    std::for_each(std::begin(vZ1), std::end(vZ1),
                  [&](double &elem) { elem = minV + (double)(rand() % gap); });
    std::for_each(std::begin(vZ2), std::end(vZ2),
                  [&](double &elem) { elem = minV + (double)(rand() % gap); });
    std::for_each(std::begin(vZ3), std::end(vZ3),
                  [&](double &elem) { elem = minV + (double)(rand() % gap); });
    std::for_each(std::begin(vZ4), std::end(vZ4),
                  [&](double &elem) { elem = minV + (double)(rand() % gap); });

    std::for_each(std::begin(vS1), std::end(vS1),
                  [&](double &elem) { elem = 0.0; });
    std::for_each(std::begin(vS2), std::end(vS2),
                  [&](double &elem) { elem = 1.0; });
    std::for_each(std::begin(vS3), std::end(vS3),
                  [&](double &elem) { elem = 2.0; });
    std::for_each(std::begin(vS4), std::end(vS4),
                  [&](double &elem) { elem = 3.0; });

    // COMPUTING THE RESULTS
    int i;
    double sum1 = 0.0, alpha1 = 1.1;
    double sum2 = 0.0, alpha2 = 2.2;
    double sum3 = 0.0, alpha3 = 3.3;
    double sum4 = 0.0, alpha4 = 4.4;
    double ONE = 1.0f;

    i = 0;
    std::for_each(std::begin(vY1), std::end(vY1), [&](double &elem) {
      elem = vZ1[i] + alpha1 * vX1[i];
      if ((i % strd) == 0) sum1 += std::abs(elem);
      i++;
    });
    //    vS1[0] = sum1;
    i = 0;
    std::for_each(std::begin(vY2), std::end(vY2), [&](double &elem) {
      elem = vZ2[i] + alpha2 * vX2[i];
      if ((i % strd) == 0) sum2 += std::abs(elem);
      i++;
    });
    //    vS2[0] = sum2;
    i = 0;
    std::for_each(std::begin(vY3), std::end(vY3), [&](double &elem) {
      elem = vZ3[i] + alpha3 * vX3[i];
      if ((i % strd) == 0) sum3 += std::abs(elem);
      i++;
    });
    //    vS3[0] = sum3;
    i = 0;
    std::for_each(std::begin(vY4), std::end(vY4), [&](double &elem) {
      elem = vZ4[i] + alpha4 * vX4[i];
      if ((i % strd) == 0) sum4 += std::abs(elem);
      i++;
    });
//    vS4[0] = sum4;

#ifdef SHOW_VALUES
    std::cout << "VECTORS BEFORE COMPUTATION" << std::endl;
    for (int i = 0; i < sizeV; i++) {
      std::cout << "Component = " << i << std::endl;
      std::cout << "vX = (" << vX1[i] << ", " << vX2[i] << ", " << vX3[i]
                << ", " << vX4[i] << ")" << std::endl;
      std::cout << "vY = (" << vY1[i] << ", " << vY2[i] << ", " << vY3[i]
                << ", " << vY4[i] << ")" << std::endl;
      std::cout << "vZ = (" << vZ1[i] << ", " << vZ2[i] << ", " << vZ3[i]
                << ", " << vZ4[i] << ")" << std::endl;
    }
#endif

    // CREATING THE SYCL QUEUE AND EXECUTOR
    cl::sycl::queue q([=](cl::sycl::exception_list eL) {
      try {
        for (auto &e : eL) {
          std::rethrow_exception(e);
        }
      } catch (cl::sycl::exception &e) {
        std::cout << " E " << e.what() << std::endl;
      } catch (...) {
        std::cout << " An exception " << std::endl;
      }
    });

    {
      cl_context clContext = q.get_context().get();
      cl_command_queue clQueue = q.get();

      cl_int err = CL_SUCCESS;

      err = clblasSetup();

      if (err != CL_SUCCESS) {
        std::cout << "Error during initialization of clBlas" << std::endl;
      }

      cl_mem bX1_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vX1.size() * sizeof(double), nullptr, &err);
      cl_mem bY1_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vY1.size() * sizeof(double), nullptr, &err);

      cl_mem bX2_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vX2.size() * sizeof(double), nullptr, &err);
      cl_mem bY2_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vY2.size() * sizeof(double), nullptr, &err);

      cl_mem bX3_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vX3.size() * sizeof(double), nullptr, &err);
      cl_mem bY3_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vY3.size() * sizeof(double), nullptr, &err);

      cl_mem bX4_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vX4.size() * sizeof(double), nullptr, &err);
      cl_mem bY4_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vY4.size() * sizeof(double), nullptr, &err);

      cl_mem bZ1_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vZ1.size() * sizeof(double), nullptr, &err);
      cl_mem bZ2_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vZ2.size() * sizeof(double), nullptr, &err);
      cl_mem bZ3_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vZ3.size() * sizeof(double), nullptr, &err);
      cl_mem bZ4_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vZ4.size() * sizeof(double), nullptr, &err);

      cl_mem bS1_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vS1.size() * sizeof(double), nullptr, &err);
      cl_mem bS2_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vS2.size() * sizeof(double), nullptr, &err);
      cl_mem bS3_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vS3.size() * sizeof(double), nullptr, &err);
      cl_mem bS4_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vS4.size() * sizeof(double), nullptr, &err);

      cl_mem scratch_cl =
          clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                         vY1.size() * sizeof(double), nullptr, &err);

      {
        err = clEnqueueWriteBuffer(clQueue, bX1_cl, CL_FALSE, 0,
                                   (vX1.size() * sizeof(double)), vX1.data(), 0,
                                   NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueWriteBuffer(clQueue, bY1_cl, CL_FALSE, 0,
                                   (vY1.size() * sizeof(double)), vY1.data(), 0,
                                   NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueWriteBuffer(clQueue, bX2_cl, CL_FALSE, 0,
                                   (vX2.size() * sizeof(double)), vX2.data(), 0,
                                   NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueWriteBuffer(clQueue, bY2_cl, CL_FALSE, 0,
                                   (vY2.size() * sizeof(double)), vY2.data(), 0,
                                   NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueWriteBuffer(clQueue, bX3_cl, CL_FALSE, 0,
                                   (vX3.size() * sizeof(double)), vX3.data(), 0,
                                   NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueWriteBuffer(clQueue, bY3_cl, CL_FALSE, 0,
                                   (vY3.size() * sizeof(double)), vY3.data(), 0,
                                   NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueWriteBuffer(clQueue, bX4_cl, CL_FALSE, 0,
                                   (vX4.size() * sizeof(double)), vX4.data(), 0,
                                   NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueWriteBuffer(clQueue, bY4_cl, CL_FALSE, 0,
                                   (vY4.size() * sizeof(double)), vY4.data(), 0,
                                   NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueWriteBuffer(clQueue, bZ1_cl, CL_FALSE, 0,
                                   (vZ1.size() * sizeof(double)), vZ1.data(), 0,
                                   NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueWriteBuffer(clQueue, bZ2_cl, CL_FALSE, 0,
                                   (vZ2.size() * sizeof(double)), vZ2.data(), 0,
                                   NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueWriteBuffer(clQueue, bZ3_cl, CL_FALSE, 0,
                                   (vZ3.size() * sizeof(double)), vZ3.data(), 0,
                                   NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueWriteBuffer(clQueue, bZ4_cl, CL_FALSE, 0,
                                   (vZ4.size() * sizeof(double)), vZ4.data(), 0,
                                   NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }
      }  // End of copy

      for (int i = 0; i < NUMBER_REPEATS; i++) {
#ifdef SHOW_TIMES
        t_start = std::chrono::steady_clock::now();
#endif
        // One copy
        {
          cl_event events[4];

          err = clblasDcopy(numE, bZ1_cl, 0, strd, bY1_cl, 0, strd, 1, &clQueue,
                            0, NULL, &events[0]);
          err = clblasDcopy(numE, bZ2_cl, 0, strd, bY2_cl, 0, strd, 1, &clQueue,
                            0, NULL, &events[1]);
          err = clblasDcopy(numE, bZ3_cl, 0, strd, bY3_cl, 0, strd, 1, &clQueue,
                            0, NULL, &events[2]);
          err = clblasDcopy(numE, bZ4_cl, 0, strd, bY4_cl, 0, strd, 1, &clQueue,
                            0, NULL, &events[3]);

          err |= clWaitForEvents(4, events);

          if (err != CL_SUCCESS) {
            std::cout << " ERROR " << err << std::endl;
          }
        }
#ifdef SHOW_TIMES
        t_stop = std::chrono::steady_clock::now();
        t0_copy = t_stop - t_start;
#endif

#ifdef SHOW_TIMES
        t_start = std::chrono::steady_clock::now();
#endif
        /* */
        // One axpy
        {
          cl_event events[4];

          err = clblasDaxpy(numE, alpha1, bX1_cl, 0, strd, bY1_cl, 0, strd, 1,
                            &clQueue, 0, NULL, &events[0]);
          err |= clblasDaxpy(numE, alpha2, bX2_cl, 0, strd, bY2_cl, 0, strd, 1,
                             &clQueue, 0, NULL, &events[1]);
          err |= clblasDaxpy(numE, alpha3, bX3_cl, 0, strd, bY3_cl, 0, strd, 1,
                             &clQueue, 0, NULL, &events[2]);
          err |= clblasDaxpy(numE, alpha4, bX4_cl, 0, strd, bY4_cl, 0, strd, 1,
                             &clQueue, 0, NULL, &events[3]);

          err |= clWaitForEvents(4, events);

          if (err != CL_SUCCESS) {
            std::cout << " ERROR " << err << std::endl;
          }
        }
/* */
#ifdef SHOW_TIMES
        t_stop = std::chrono::steady_clock::now();
        t0_axpy = t_stop - t_start;
#endif

#ifdef SHOW_TIMES
        t_start = std::chrono::steady_clock::now();
#endif
        // One add
        {
          cl_event events[4];
          err = clblasDasum(numE,
                            //              bY1_cl, 0,
                            //              bS1_cl, 0, 1,
                            bS1_cl, 0, bY1_cl, 0, strd, scratch_cl, 1, &clQueue,
                            0, NULL, &events[0]);
          err = clblasDasum(numE,
                            //              bY2_cl, 0,
                            //              bS2_cl, 0, 1,
                            bS2_cl, 0, bY2_cl, 0, strd, scratch_cl, 1, &clQueue,
                            0, NULL, &events[1]);
          err = clblasDasum(numE,
                            //              bY3_cl, 0,
                            //              bS3_cl, 0, 1,
                            bS3_cl, 0, bY3_cl, 0, strd, scratch_cl, 1, &clQueue,
                            0, NULL, &events[2]);
          err = clblasDasum(numE,
                            //              bY4_cl, 0,
                            //              bS4_cl, 0, 1,
                            bS4_cl, 0, bY4_cl, 0, strd, scratch_cl, 1, &clQueue,
                            0, NULL, &events[3]);

          err |= clWaitForEvents(4, events);

          if (err != CL_SUCCESS) {
            std::cout << __LINE__ << ": ERROR " << err << std::endl;
          }
        }  // End of copy
#ifdef SHOW_TIMES
        t_stop = std::chrono::steady_clock::now();
        t0_add = t_stop - t_start;
#endif
      }

      {
        err = clEnqueueReadBuffer(clQueue, bX1_cl, CL_FALSE, 0,
                                  (vX1.size() * sizeof(double)), vX1.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bY1_cl, CL_FALSE, 0,
                                  (vY1.size() * sizeof(double)), vY1.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bX2_cl, CL_FALSE, 0,
                                  (vX2.size() * sizeof(double)), vX2.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bY2_cl, CL_FALSE, 0,
                                  (vY2.size() * sizeof(double)), vY2.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bX3_cl, CL_FALSE, 0,
                                  (vX3.size() * sizeof(double)), vX3.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bY3_cl, CL_FALSE, 0,
                                  (vY3.size() * sizeof(double)), vY3.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bX4_cl, CL_FALSE, 0,
                                  (vX4.size() * sizeof(double)), vX4.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bY4_cl, CL_FALSE, 0,
                                  (vY4.size() * sizeof(double)), vY4.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bZ1_cl, CL_FALSE, 0,
                                  (vZ1.size() * sizeof(double)), vZ1.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bZ2_cl, CL_FALSE, 0,
                                  (vZ2.size() * sizeof(double)), vZ2.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bZ3_cl, CL_FALSE, 0,
                                  (vZ3.size() * sizeof(double)), vZ3.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bZ4_cl, CL_FALSE, 0,
                                  (vZ4.size() * sizeof(double)), vZ4.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bS1_cl, CL_FALSE, 0,
                                  (vS1.size() * sizeof(double)), vS1.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bS2_cl, CL_FALSE, 0,
                                  (vS2.size() * sizeof(double)), vS2.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bS3_cl, CL_FALSE, 0,
                                  (vS3.size() * sizeof(double)), vS3.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

        err = clEnqueueReadBuffer(clQueue, bS4_cl, CL_FALSE, 0,
                                  (vS4.size() * sizeof(double)), vS4.data(), 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
          std::cout << " Error copying to device " << err << std::endl;
        }

      }  // End of enqueue
      clFinish(clQueue);

      clReleaseMemObject(bX1_cl);
      clReleaseMemObject(bY1_cl);
      clReleaseMemObject(bX2_cl);
      clReleaseMemObject(bY2_cl);
      clReleaseMemObject(bX3_cl);
      clReleaseMemObject(bY3_cl);
      clReleaseMemObject(bX4_cl);
      clReleaseMemObject(bY4_cl);
      clReleaseMemObject(bZ1_cl);
      clReleaseMemObject(bZ2_cl);
      clReleaseMemObject(bZ3_cl);
      clReleaseMemObject(bZ4_cl);
      clReleaseMemObject(bS1_cl);
      clReleaseMemObject(bS2_cl);
      clReleaseMemObject(bS3_cl);
      clReleaseMemObject(bS4_cl);

      clblasTeardown();
    }

#ifdef SHOW_VALUES
    std::cout << "VECTORS AFTER  COMPUTATION" << std::endl;
    for (int i = 0; i < sizeV; i++) {
      std::cout << "Component = " << i << std::endl;
      std::cout << "vX = (" << vX1[i] << ", " << vX2[i] << ", " << vX3[i]
                << ", " << vX4[i] << ")" << std::endl;
      std::cout << "vY = (" << vY1[i] << ", " << vY2[i] << ", " << vY3[i]
                << ", " << vY4[i] << ")" << std::endl;
      std::cout << "vZ = (" << vZ1[i] << ", " << vZ2[i] << ", " << vZ3[i]
                << ", " << vZ4[i] << ")" << std::endl;
    }
#endif

#ifdef SHOW_TIMES
    // COMPUTATIONAL TIMES
    std::cout << "t_copy, " << t0_copy.count() << std::endl;
    //    std::cout <<   "t_copy --> (" << t0_copy.count() << ", " <<
    //    t1_copy.count()
    //                          << ", " << t2_copy.count() << ", " <<
    //                          t3_copy.count() << ")" << std::endl;
    std::cout << "t_axpy, " << t0_axpy.count() << std::endl;
    //    std::cout <<   "t_axpy --> (" << t0_axpy.count() << ", " <<
    //    t1_axpy.count()
    //                          << ", " << t2_axpy.count() << ", " <<
    //                          t3_axpy.count() << ")" << std::endl;
    std::cout << "t_add, " << t0_add.count() << std::endl;
//    std::cout <<   "t_add  --> (" << t0_add.count()  << ", " << t1_add.count()
//                          << ", " << t2_add.count()  << ", " << t3_add.count()
//                          << ")" << std::endl;
//
#endif

    // ANALYSIS OF THE RESULTS
    double res;

    for (i = 0; i < 0; i++) {
      res = vS1[i];
#ifdef SHOW_VALUES
      std::cout << "VALUES!! --> res = " << res << " , i = " << i
                << " , sum = " << sum1 << " , err = " << res - sum1
                << std::endl;
#endif  //  SHOW_VALUES
      if (std::abs((res - sum1) / res) > ERROR_ALLOWED) {
        std::cout << "VALUES!! --> res = " << res << " , i = " << i
                  << " , sum = " << sum1 << " , err = " << res - sum1
                  << std::endl;
        returnVal += 2 * i;
      }
    }

    for (i = 0; i < 0; i++) {
      res = vS2[i];
#ifdef SHOW_VALUES
      std::cout << "VALUES!! --> res = " << res << " , i = " << i
                << " , sum = " << sum2 << " , err = " << res - sum2
                << std::endl;
#endif  //  SHOW_VALUES
      if (std::abs((res - sum2) / res) > ERROR_ALLOWED) {
        std::cout << "VALUES!! --> res = " << res << " , i = " << i
                  << " , sum = " << sum2 << " , err = " << res - sum2
                  << std::endl;
        returnVal += 20 * i;
      }
    }

    for (i = 0; i < 0; i++) {
      res = vS3[i];
#ifdef SHOW_VALUES
      std::cout << "VALUES!! --> res = " << res << " , i = " << i
                << " , sum = " << sum3 << " , err = " << res - sum3
                << std::endl;
#endif  //  SHOW_VALUES
      if (std::abs((res - sum3) / res) > ERROR_ALLOWED) {
        std::cout << "VALUES!! --> res = " << res << " , i = " << i
                  << " , sum = " << sum3 << " , err = " << res - sum3
                  << std::endl;
        returnVal += 200 * i;
      }
    }

    for (i = 0; i < 0; i++) {
      res = vS4[i];
#ifdef SHOW_VALUES
      std::cout << "VALUES!! --> res = " << res << " , i = " << i
                << " , sum = " << sum4 << " , err = " << res - sum4
                << std::endl;
#endif  //  SHOW_VALUES
      if (std::abs((res - sum4) / res) > ERROR_ALLOWED) {
        std::cout << "VALUES!! --> res = " << res << " , i = " << i
                  << " , sum = " << sum4 << " , err = " << res - sum4
                  << std::endl;
        returnVal += 2000 * i;
      }
    }
  }

  return returnVal;
}
