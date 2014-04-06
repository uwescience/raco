from osutils import Chdir
import subprocess

chdir = Chdir("c_test_environment")

subprocess.check_call(['clang', '-std=c++11', 'basic.cpp', '-o', 'basic.exe'])
subprocess.check_call(['./basic.exe'])

