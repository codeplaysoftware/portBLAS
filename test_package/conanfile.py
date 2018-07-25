from conans import ConanFile, tools, CMake
import os


class SYCLBLASpPackageTest(ConanFile):
    generators = "cmake_paths"
    
    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def test(self):
        self.run("./app")
