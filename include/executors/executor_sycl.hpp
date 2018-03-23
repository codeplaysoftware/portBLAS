/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2016 Codeplay Software Limited
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
 *  @filename executor_sycl.hpp
 *
 **************************************************************************/

#ifndef EXECUTOR_SYCL_HPP
#define EXECUTOR_SYCL_HPP

#include <stdexcept>

#include <CL/sycl.hpp>

#include <executors/blas1_tree_executor.hpp>
#include <executors/blas2_tree_executor.hpp>
#include <executors/blas3_tree_executor.hpp>
#include <executors/executor_base.hpp>
#include <operations/blas1_trees.hpp>
#include <operations/blas2_trees.hpp>
#include <operations/blas3_trees.hpp>
#include <queue/queue_sycl.hpp>
#include <views/view_sycl.hpp>

namespace blas {

/*!using_shared_mem.
 * @brief Indicates whether if the kernel uses shared memory or not.
 This is a work-around for the following enum class.
 //enum class using_shared_mem { enabled, disabled };
 When the member of enum class is used as a template parameters of the
 kernel functor, our duplicator cannot capture those elements.
 One way to solve it is to replace enum with namespace and replace each member
 of enum with static if. the other way is changing the order of stub file which
 is not recommended.
 */
namespace using_shared_mem {
static const int enabled = 0;
static const int disabled = 1;
};  // namespace using_shared_mem

/*!
@brief Template alias for a local accessor.
@tparam valueT Value type of the accessor.
*/
template <typename valueT>
using local_accessor_t =
    cl::sycl::accessor<valueT, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::local>;

/*!
@brief A struct for containing a local accessor if shared memory is enabled.
Non-specialised case for usingSharedMem == enabled, which contains a local
accessor.
@tparam valueT Value type of the local accessor.
@tparam usingSharedMem Enum class specifying whether shared memory is enabled.
*/
template <typename valueT, int usingSharedMem>
struct shared_mem {
  /*!
  @brief Constructor that creates a local accessor from a size and a SYCL
  command group handler.
  @param size Size in elements of the local accessor.
  @param cgh SYCL command group handler.
  */
  shared_mem(size_t size, cl::sycl::handler &cgh)
      : localAcc(cl::sycl::range<1>(size), cgh) {}

  /*!
  @brief Subscirpt operator that forwards on to the local accessor subscript
  operator.
  @param id SYCL id.
  @return Reference to an element of the local accessor.
  */
  valueT &operator[](cl::sycl::id<1> id) { return localAcc[id]; }

  /*!
  @brief Local accessor.
  */
  local_accessor_t<valueT> localAcc;
};

/*!
@brief A struct for containing a local accessor if shared memory is enabled.
Specialised case for usingSharedMem == disabled, which does not contain a local
accessor.
@tparam valueT Value type of the accessor.
*/
template <typename valueT>
struct shared_mem<valueT, using_shared_mem::disabled> {
  /*!
  @brief Constructor that does nothing.
  @param size Size in elements of the local accessor.
  @param cgh SYCL command group handler.
  */
  shared_mem(size_t size, cl::sycl::handler &cgh) {}
};

/*!
@brief Template struct defining the value type of shared memory if enabled.
Non-specialised case for usingSharedMem == enabled, which defines the type as
the value type of the tree.
@tparam usingSharedMemory Enum class specifying whether shared memory is
enabled.
*/
template <int usingSharedMem, typename treeT>
struct shared_mem_type {
  using type = typename blas::Evaluate<treeT>::value_type;
};

/*!
@brief Template struct defining the value type of shared memory if enabled.
Specialised case for usingSharedMem == disabled, which defines the type as void.
@tparam usingSharedMemory Enum class specifying whether shared memory is
enabled.
*/
template <typename treeT>
struct shared_mem_type<using_shared_mem::disabled, treeT> {
  using type = void;
};

/*!
@brief Template struct for containing an eval function, which uses shared memory
if enabled. Non-specialised case for usingSharedMem == enabled, which calls eval
on the tree with the shared_memory as well as an index.
@tparam usingSharedMem Enum class specifying whether shared memory is enabled.
@tparam treeT Type of the tree.
@tparam sharedMemT Value type of the shared memory.
*/
template <int usingSharedMem, typename treeT, typename sharedMemT>
struct tree {
  /*!
  @brief Static function that calls eval on a tree, passing the shared_memory
  object and the index.
  @param tree Tree object.
  @param scratch Shared memory object.
  @param index SYCL nd_item.
  */
  static void eval(treeT &tree, shared_mem<sharedMemT, usingSharedMem> scratch,
                   cl::sycl::nd_item<1> index) {
    tree.eval(scratch, index);
  }
};

/*! tree.
@brief Template struct for containing an eval function, which uses shared memory
if enabled. Specialised case for usingSharedMem == false, which calls eval on
the tree with the index only.
@tparam usingSharedMem Enum class specifying whether shared memory is enabled.
@tparam treeT Type of the tree.
@tparam sharedMemT Value type of the shared memory.
*/
template <typename treeT, typename sharedMemT>
struct tree<using_shared_mem::disabled, treeT, sharedMemT> {
  /*!
  @brief Static function that calls eval on a tree, passing only the index.
  @param tree Tree object.
  @param scratch Shared memory object.
  @param index SYCL nd_item.
  */
  static void eval(treeT &tree,
                   shared_mem<sharedMemT, using_shared_mem::disabled> scratch,
                   cl::sycl::nd_item<1> index) {
    if ((index.get_global(0) < tree.getSize())) {  // FIXME:: this should move
                                                   // to the tree not the root
      tree.eval(index);
    }
  }
};

/*! execute_tree.
@brief the functor for executing a tree in SYCL.
@tparam int usingSharedMem  specifying whether shared memory is enabled.
@tparam Evaluator Type of the tree.
@tparam value_type type of elements in shared memory.
@tparam SharedMem shared memory type (local accessor type).
@param scratch_ shared memory object (local accessor).
@param evaluator_ Tree object.
*/
template <int usingSharedMem, typename Evaluator, typename SharedMem,
          typename value_type>
struct ExecTreeFunctor {
  SharedMem scratch;
  Evaluator evaluator;
  ExecTreeFunctor(SharedMem scratch_, Evaluator evaluator_)
      : scratch(scratch_), evaluator(evaluator_) {}
  void operator()(cl::sycl::nd_item<1> i) {
    tree<usingSharedMem, Evaluator, value_type>::eval(evaluator, scratch, i);
  }
};
/*! execute_tree.
@brief Static function for executing a tree in SYCL.
@tparam int usingSharedMem specifying whether shared memory is enabled.
@tparam Tree Type of the tree.
@param q_ SYCL queue.
@param t Tree object.
@param _localSize Local work group size.
@param _globalSize Global work size.
@param _shMem Size in elements of the shared memory (should be zero if
usingSharedMem == false).
*/
template <int usingSharedMem, typename Tree>
static void execute_tree(cl::sycl::queue q_, Tree t, size_t _localSize,
                         size_t _globalSize, size_t _shMem) {
  using value_type = typename shared_mem_type<usingSharedMem, Tree>::type;

  auto localSize = _localSize;
  auto globalSize = _globalSize;
  auto shMem = _shMem;

  auto cg1 = [=](cl::sycl::handler &h) mutable {
    auto nTree = blas::make_accessor(t, h);

    auto scratch = shared_mem<value_type, usingSharedMem>(shMem, h);

    cl::sycl::nd_range<1> gridConfiguration = cl::sycl::nd_range<1>{
        cl::sycl::range<1>{globalSize}, cl::sycl::range<1>{localSize}};
    h.parallel_for(
        gridConfiguration,
        ExecTreeFunctor<usingSharedMem, decltype(nTree), decltype(scratch),
                        value_type>(scratch, nTree));
  };

  q_.submit(cg1);
}

/*! Executor<SYCL>.
 * @brief Executes an Expression Tree using SYCL.
 */
template <>
class Executor<SYCL> {
  // This should be added up on request
  // bool is_pointer_mapper_owner;
  Queue_Interface<SYCL> q_interface;

 public:
  template <typename T>
  using ContainerT = bufferT<T>;
  /*!
   * @brief Constructs a SYCL executor using the given queue.
   * @param q A SYCL queue.
   */
  Executor(cl::sycl::queue q) : q_interface(q){};

  // FIXME:The pointermapper constructor is deleted (later on it should be
  // added)
  /*Executor(const Queue_Interface<SYCL> &q_interface)
      : q_interface(q_interface) {}*/

  cl::sycl::queue sycl_queue() const { return q_interface.sycl_queue(); }

  inline Queue_Interface<SYCL>::device_type get_device_type() {
    return q_interface.get_device_type();
  }

  inline bool has_local_memory() const {
    return q_interface.has_local_memory();
  }
  template <typename T>
  inline T *allocate(size_t num_elements) const {
    return q_interface.template allocate<T>(num_elements);
  }

  template <typename T>
  inline void deallocate(T *p) const {
    q_interface.deallocate(p);
  }
  /*
  @brief this class is to return the dedicated buffer to the user
  @ tparam T is the type of the pointer
  @tparam bufferT<T> is the type of the buffer points to the data. on the host
  side buffer<T> and T are the same
  */
  template <typename T>
  inline bufferT<T> get_buffer(T *ptr) const {
    return q_interface.get_buffer(ptr);
  }
  /*
  @brief this function is to get the offset from the actual pointer
  @tparam T is the type of the pointer
  */
  template <typename T>
  inline ptrdiff_t get_offset(T *ptr) const {
    return q_interface.get_offset(ptr);
  }
  /*  @brief Copying the data back to device
      @tparam T is the type of the data
      @param src is the host pointer we want to copy from.
      @param dst is the device pointer we want to copy to.
      @param size is the number of elements to be copied
  */
  template <typename T>
  inline void copy_to_device(T *src, T *dst, size_t size) {
    q_interface.copy_to_device(src, dst, size);
  }
  /*  @brief Copying the data back to device
      @tparam T is the type of the data
      @param src is the device pointer we want to copy from.
      @param dst is the host pointer we want to copy to.
      @param size is the number of elements to be copied
  */
  template <typename T>
  inline void copy_to_host(T *src, T *dst, size_t size) {
    q_interface.copy_to_host(src, dst, size);
  }

  /*!
   * @brief Executes the tree without defining required shared memory.
   */
  template <typename Tree>
  inline void execute(Tree t) {
    const auto localSize = 128;
    auto _N = t.getSize();
    auto nWG = (_N + localSize - 1) / localSize;
    auto globalSize = nWG * localSize;

    execute_tree<using_shared_mem::disabled>(q_interface.sycl_queue(), t,
                                             localSize, globalSize, 0);
  };

  /*!
   * @brief Executes the tree fixing the localSize but without defining required
   * shared memory.
   */
  template <typename Tree>
  void execute(Tree t, size_t localSize) {
    auto _N = t.getSize();
    auto nWG = (_N + localSize - 1) / localSize;
    auto globalSize = nWG * localSize;

    execute_tree<using_shared_mem::disabled>(q_interface.sycl_queue(), t,
                                             localSize, globalSize, 0);
  };

  /*!
   * @brief Executes the tree with specific local, global and shared memory
   * values.
   */
  template <typename Tree>
  void execute(Tree t, size_t _localSize, size_t _globalSize, size_t _shMem) {
    auto localSize = _localSize;
    auto globalSize = _globalSize;
    auto shMem = _shMem;
    execute_tree<using_shared_mem::enabled>(q_interface.sycl_queue(), t,
                                            localSize, globalSize, shMem);
  }

  /*!
   * @brief Applies a reduction to a tree.
   */
  template <typename Tree>
  void reduce(Tree t) {
    using oper_type = typename blas::Evaluate<Tree>::oper_type;
    using input_type = typename blas::Evaluate<Tree>::input_type;
    using cont_type = typename blas::Evaluate<Tree>::cont_type;
    using LHS_type = typename blas::Evaluate<Tree>::LHS_type;
    auto _N = t.getSize();
    auto localSize = t.blqS;
    // IF THERE ARE ENOUGH ELEMENTS, EACH BLOCK PROCESS TWO BLOCKS OF
    // ELEMENTS THEREFORE, 2*GLOBALSIZE ELEMENTS ARE PROCESSED IN A STEP
    // MOREOVER, A LOOP ALLOWS TO REPEAT THE PROCESS UNTIL
    // ALL THE ELEMENTS ARE PROCESSED
    auto nWG = (t.grdS + (2 * localSize) - 1) / (2 * localSize);
    auto lhs = t.l;
    auto rhs = t.r;

    // Two accessors to local memory
    auto sharedSize = ((nWG < localSize) ? localSize : nWG);
    cont_type shMem1(sharedSize);
    auto opShMem1 = LHS_type(shMem1, 0, 1, sharedSize);
    cont_type shMem2(sharedSize);
    auto opShMem2 = LHS_type(shMem2, 0, 1, sharedSize);

    bool frst = true;
    bool even = false;
    do {
      auto globalSize = nWG * localSize;
      if (frst) {
        // THE FIRST CASE USES THE ORIGINAL BINARY/TERNARY FUNCTION
        auto localTree = input_type(((nWG == 1) ? lhs : opShMem1), rhs,
                                    localSize, globalSize);
        execute_tree<using_shared_mem::enabled>(q_interface.sycl_queue(),
                                                localTree, localSize,
                                                globalSize, sharedSize);
      } else {
        // THE OTHER CASES ALWAYS USE THE BINARY FUNCTION
        auto localTree = blas::AssignReduction<oper_type, LHS_type, LHS_type>(
            ((nWG == 1) ? lhs : (even ? opShMem2 : opShMem1)),
            (even ? opShMem1 : opShMem2), localSize, globalSize);
        execute_tree<using_shared_mem::enabled>(q_interface.sycl_queue(),
                                                localTree, localSize,
                                                globalSize, sharedSize);
      }
      _N = nWG;
      nWG = (_N + (2 * localSize) - 1) / (2 * localSize);
      frst = false;
      even = !even;
    } while (_N > 1);
  };

  /*!
   * @brief Applies a reduction to a tree, receiving a scratch buffer.
   */
  template <typename Tree, typename Scratch>
  void reduce(Tree t, Scratch scr) {
    using oper_type = typename blas::Evaluate<Tree>::oper_type;
    using input_type = typename blas::Evaluate<Tree>::input_type;
    using cont_type = typename blas::Evaluate<Tree>::cont_type;
    using LHS_type = typename blas::Evaluate<Tree>::LHS_type;
    auto _N = t.getSize();
    auto localSize = t.blqS;
    // IF THERE ARE ENOUGH ELEMENTS, EACH BLOCK PROCESS TWO BLOCKS OF
    // ELEMENTS THEREFORE, 2*GLOBALSIZE ELEMENTS ARE PROCESSED IN A STEP
    // MOREOVER, A LOOP ALLOWS TO REPEAT THE PROCESS UNTIL
    // ALL THE ELEMENTS ARE PROCESSED
    auto nWG = (t.grdS + (2 * localSize) - 1) / (2 * localSize);
    auto lhs = t.l;
    auto rhs = t.r;

    // Two accessors to local memory
    auto sharedSize = ((nWG < localSize) ? localSize : nWG);
    auto opShMem1 = LHS_type(scr, 0, 1, sharedSize);
    auto opShMem2 = LHS_type(scr, sharedSize, 1, sharedSize);

    bool frst = true;
    bool even = false;
    do {
      auto globalSize = nWG * localSize;
      if (frst) {
        // THE FIRST CASE USES THE ORIGINAL BINARY/TERNARY FUNCTION
        auto localTree = input_type(((nWG == 1) ? lhs : opShMem1), rhs,
                                    localSize, globalSize);
        execute_tree<using_shared_mem::enabled>(q_interface.sycl_queue(),
                                                localTree, localSize,
                                                globalSize, sharedSize);
      } else {
        // THE OTHER CASES ALWAYS USE THE BINARY FUNCTION
        auto localTree = blas::AssignReduction<oper_type, LHS_type, LHS_type>(
            ((nWG == 1) ? lhs : (even ? opShMem2 : opShMem1)),
            (even ? opShMem1 : opShMem2), localSize, globalSize);
        execute_tree<using_shared_mem::enabled>(q_interface.sycl_queue(),
                                                localTree, localSize,
                                                globalSize, sharedSize);
      }
      _N = nWG;
      nWG = (_N + (2 * localSize) - 1) / (2 * localSize);
      frst = false;
      even = !even;
    } while (_N > 1);
  };

  template <bool Conds, int T1, int T2>
  struct Choose_policy {
    static const int type = T1;
  };

  template <int T1, int T2>
  struct Choose_policy<false, T1, T2> {
    static const int type = T2;
  };

  template <typename Gemm>
  inline void gemm_executor(Gemm gemm_tree) {
    auto rng = Gemm::get_nd_range(gemm_tree.m, gemm_tree.n);
    execute_tree<Choose_policy<Gemm::version == 19, using_shared_mem::enabled,
                               using_shared_mem::disabled>::type>(
        q_interface.sycl_queue(), gemm_tree, rng.get_local()[0],
        rng.get_global()[0], Gemm::scratch_size);
  }
};

}  // namespace blas

#endif  // EXECUTOR_SYCL_HPP
