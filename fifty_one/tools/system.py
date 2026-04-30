import os
import platform
import socket

#=====================================================================================================

#------------------------------------------------------------------------------------------
#------------------------------
# Function to clear the console screen
#------------------------------
def clear_screen() -> None:
    """
    Clear the console screen depending on the operating system.

    Uses:
        - 'cls' on Windows
        - 'clear' on Unix-based systems (Linux / macOS)
    """
    os.system("cls" if os.name == "nt" else "clear")
    
def est_windows():
    return platform.system().lower() == "windows" 

def est_linux():
    return platform.system().lower() == "linux"

# ------------------------------
# Function to find a free port
# ------------------------------
def get_free_port() -> int:
    """
    Returns an available port on the local machine.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Let OS assign a free port
        return s.getsockname()[1]
    
