import os
import joblib
import sys
sys.path.append(os.getcwd())

class IO_Manager:
    prototypes_dict :dict =None
    rw_kernels_dict :dict =None
    @staticmethod
    def get_calculator():
        pass

    @staticmethod
    def get_prototype_selector():
        if IO_Manager.prototypes_dict is None:
            try:
                IO_Manager.prototypes_dict = joblib.load("prototypes.joblib")
            except FileNotFoundError:
                IO_Manager.prototypes_dict = {}
        return IO_Manager.prototypes_dict

    @staticmethod
    def add_prototype_selector(key, value):
        if IO_Manager.prototypes_dict is None:
            IO_Manager.prototypes_dict = {}
        IO_Manager.prototypes_dict[key] = value

    @staticmethod
    def save_prototype_selector():
        if IO_Manager.prototypes_dict is not None:
            try:
                joblib.dump(IO_Manager.prototypes_dict, "prototypes.joblib")
            except Exception as e:
                print(f"Error saving prototypes: {e}")
    @staticmethod
    def save_rw_kernel_matrix(key, matrix):
        if IO_Manager.rw_kernels_dict is None:
            IO_Manager.rw_kernels_dict = {}
        IO_Manager.rw_kernels_dict[key] = matrix
    @staticmethod
    def get_rw_kernel_matrix(key):
        if IO_Manager.rw_kernels_dict is None:
            return None
        else:
            return IO_Manager.rw_kernels_dict.get(key, None)
        

                