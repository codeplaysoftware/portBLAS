from conans import ConanFile, CMake

class SYCLBLASConan(ConanFile):
    name = "sycl-blas"
    version = "0.1"
    requires = "computecpp/0.9.1@codeplay/testing", "openblas/0.2.20@conan/stable"
    build_requires= "gtest/1.8.0@bincrafters/stable"
    generators = "cmake_paths"
    no_copy_source = True
    exports_sources = "*", "!build*/*", "!.git/*"

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
        cmake.test()
        cmake.install()

    def package_id(self):
        self.info.header_only()
