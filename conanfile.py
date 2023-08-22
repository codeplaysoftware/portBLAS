from conans import ConanFile, tools, CMake, RunEnvironment
from conans.errors import ConanInvalidConfiguration, ConanException
import os


class PortBlasConan(ConanFile):
    name = "portBLAS"
    version = "1.0"
    settings = "os", "compiler", "build_type", "arch"
    description = "An implementation of BLAS using the SYCL open standard for acceleration on OpenCL devices"
    url = "https://github.com/codeplaysoftware/portBLAS"
    license = "Apache-2.0"
    author = "Codeplay Software Ltd."
    topics = ('sycl', 'blas')

    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "acl_backend": ["neon", "opencl"],
        "build_acl_benchmarks": [True, False],
        "build_benchmarks": [True, False],
        "build_clblast_benchmarks": [True, False],
        "build_expression_tests": [True, False],
        "build_testing": [True, False],
        "sycl_target": "ANY",
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "acl_backend": "opencl",
        "build_acl_benchmarks": False,
        "build_benchmarks": False,
        "build_clblast_benchmarks": False,
        "build_expression_tests": False,
        "build_testing": False,
        "khronos-opencl-icd-loader:shared": True,
        "clblast:shared": True,
        "sycl_target": "spirv64"
    }

    scm = {
        "type": "git",
        "url": "auto",
        "revision": "auto",
        "submodule": "recursive",
    }

    generators = "cmake"

    def dep(self, package, fallback_user="_", fallback_channel="_"):
        """
        Helper function to switch between internal package forks and community packages
        """
        try:
            if self.user and self.channel:
                return "%s@%s/%s" % (package, self.user, self.channel)
        except ConanException:
            pass
        return "%s@%s/%s" % (package, fallback_user, fallback_channel)

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if not self.options.build_benchmarks:
            if self.options.build_acl_benchmarks:
                raise ConanInvalidConfiguration("build_acl_benchmarks requires build_benchmarks")
            if self.options.build_clblast_benchmarks:
                raise ConanInvalidConfiguration("build_clblast_benchmarks requires build_benchmarks")
        if not self.options.build_testing:
            if self.options.build_expression_tests:
                raise ConanInvalidConfiguration("build_expression_tests requires build_testing")

    def build_requirements(self):
        def build_dep(package, fallback_user="_", fallback_channel="_"):
            return self.build_requires(self.dep(package, fallback_user, fallback_channel))
        if self.options.build_benchmarks:
            build_dep("benchmark/1.5.0")
        if self.options.build_acl_benchmarks:
            build_dep("computelibrary/19.08", "mmha", "stable")
        if self.options.build_clblast_benchmarks:
            build_dep("clblast/1.5.0", "mmha", "stable")
        if self.options.build_testing:
            build_dep("gtest/1.10.0")
        if self.options.build_testing or self.options.build_benchmarks:
            build_dep("clara/1.1.5", "bincrafters", "stable")
            build_dep("openblas/0.3.7", "mmha", "stable")

    def requirements(self):
        self.requires(self.dep("khronos-opencl-icd-loader/20191007", "bincrafters", "stable"), override=True)
        self.requires(self.dep("khronos-opencl-headers/20190806", "bincrafters", "stable"), override=True)

    def imports(self):
        tools.get(
            "https://computecpp.codeplay.com/downloads/computecpp-ce/1.1.6/ubuntu-16.04-64bit.tar.gz"
        )

    _cmake = None

    @property
    def cmake(self):
        if self._cmake is None:
            self._cmake = CMake(self)
            ccp_path = os.path.join(self.build_folder,
                                    "ComputeCpp-CE-1.1.6-Ubuntu-16.04-x86_64")
            clblast_benchmarks = self.options.build_clblast_benchmarks

            config = {
                "ACL_BACKEND": str(self.options.acl_backend).upper(),
                "BLAS_ENABLE_BENCHMARK": self.options.build_benchmarks,
                "BLAS_ENABLE_TESTING": self.options.build_testing,
                "BLAS_VERIFY_BENCHMARK": self.options.build_benchmarks,
                "BUILD_ACL_BENCHMARKS": self.options.build_acl_benchmarks,
                "BUILD_CLBLAST_BENCHMARKS": clblast_benchmarks,
                "COMPUTECPP_BITCODE": self.options.sycl_target,
                "ComputeCpp_DIR": ccp_path,
                "ENABLE_EXPRESSION_TESTS": self.options.build_expression_tests,
            }

            self._cmake.definitions.update(config)
            with tools.environment_append(RunEnvironment(self).vars):
                self._cmake.configure()
        return self._cmake

    def build(self):
        with tools.environment_append(RunEnvironment(self).vars):
            self.cmake.build()
            if self.options.build_testing:
                self.cmake.test()

    def package(self):
        self.cmake.install()

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)

    def package_id(self):
        del self.info.options.build_testing
        del self.info.options.build_expression_tests
        del self.info.options.build_benchmarks
        del self.info.options.build_acl_benchmarks
        del self.info.options.build_clblast_benchmarks
        del self.info.options.acl_backend
