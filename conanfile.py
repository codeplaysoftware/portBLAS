from conans import ConanFile, CMake, tools


class SyclblasConan(ConanFile):
    name = "sycl-blas"
    version = "0.1"
    license = "Apache 2.0"
    author = "Codeplay Software Ltd."
    url = "https://github.com/codeplaysoftware/sycl-blas"
    description = "An implementation of BLAS using the SYCL open standard for acceleration on OpenCL devices"
    topics = ("sycl", "gpu", "blas", "c++")
    options = {
        "shared": [True, False],
        "build_testing": [True, False],
        "build_benchmarks": [True, False]
    }
    default_options = {
        "shared": True,
        "build_testing": False,
        "build_benchmarks": False
    }

    generators = "cmake_paths", "virtualenv", "virtualrunenv"
    exports_sources = ".clang-*", "CMakeLists.txt", "benchmark/*", "cmake/*", "external/*", "include/*", "src/*", "test/*"
    no_copy_source = True

    def build_requirements(self):
        if self.options.build_testing:
            self.build_requires("gtest/1.9.0-master.20190523@mmha/stable")
            self.build_requires("openblas/0.3.5@conan/stable")
            self.build_requires("clblast/1.5.0@mmha/stable")
        if self.options.build_benchmarks:
            self.build_requires("google-benchmark/1.4.1@mpusz/stable")
        if self.options.build_testing or self.options.build_benchmarks:
            self.build_requires("clara/1.1.5@bincrafters/stable")

    def build(self):
        cmake = CMake(self)
        cmake.definitions["BUILD_TESTING"] = self.options.build_testing
        cmake.definitions["BUILD_BENCHMARKS"] = self.options.build_benchmarks
        cmake.configure()
        cmake.build()
        if self.options.build_testing:
            cmake.test()
        cmake.install()

    def package_id(self):
        del self.info.options.build_testing
        del self.info.options.build_benchmarks
