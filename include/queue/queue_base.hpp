#ifndef QUEUE_BASE_HPP
#define QUEUE_BASE_HPP
class Sequential {};
class Parallel {};
class SYCL {};
namespace blas {
template <class ExecutionPolicy>
class Queue_Interface {
  Queue_Interface() = delete;
  /*
  @brief This class is to determine whether or not the underlying device has
  dedicated shared memory
  */
  inline bool has_local_memory() const;
  /*
   @brief This class is used to allocated the a regin of memory on the device
   @tparam T the type of the pointer
   @param num_elements number of elements of the buffer
  */
  template <typename T>
  inline T *allocate(size_t num_elements) const;
  /*
  @brief this class is to deallocate the provided region of memory on the device
  @tparam T the type of the pointer
  @param p the pointer to be deleted
   */
  template <typename T>
  inline void deallocate(T *p) const;
};  // namespace blastemplate<classExecutionPolicy>classQueue_Interface
}  // namespace blas
#endif
