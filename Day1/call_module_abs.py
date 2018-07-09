#from module1 import *
import importlib.util
spec = importlib.util.spec_from_file_location("module1", "D:/0_Training/AI/python tutorial/0_code/lib/module1.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)

print(foo.test1(10))
