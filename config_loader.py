# config loader
import os
from configparser import ConfigParser
import sys
sys.path.append(os.path.join(os.getcwd(),'configs'))
def get_conifg_param(module=None,parametername=None,type=None):
    CONFIG = ConfigParser()
    parameters = CONFIG.read(os.path.join(os.getcwd(), "configs", "Config.ini"))
    # parameters=CONFIG.read(r"C:\Users\simon\OneDrive\Desktop\Semester 7\BA\Code\getting Started\configs\Config.ini")
    if parameters == []:
        raise FileNotFoundError("The config file 'config.ini' was not found in the current directory.")
    if parametername is None:
        return CONFIG
    elif parametername == 'DEBUG':
        if module is None:
            return CONFIG.getboolean('GLOBAL', 'DEBUG')
        else:
            return CONFIG.getboolean(module, 'DEBUG') or CONFIG.getboolean('GLOBAL', 'DEBUG')
    if module is None:
        module = 'GLOBAL'   
    else:
        if CONFIG.has_option(module, parametername):
            if type is None or type == 'str':
                return CONFIG.get(module, parametername)
            elif type == 'int':
                return CONFIG.getint(module, parametername)
            elif type == 'float':
                return CONFIG.getfloat(module, parametername)
            elif type == 'bool':
                return CONFIG.getboolean(module, parametername)
            else:
                print(f"Unknown type '{type}' for parameter '{parametername}' in module '{module}'. Returning as string.")
                return CONFIG.get(module, parametername)
        else:
            if CONFIG.has_section(module):
                raise ValueError(f"the Module '{module}' does not have the parameter '{parametername}'")
            else:
                raise ValueError(f"the Module '{module}' does not exist in the config file")
