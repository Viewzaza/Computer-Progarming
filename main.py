"""
Main entry point for the Orbit Propagator Calculator application.
"""

import tkinter as tk
from src.gui import OrbitPropagatorGUI

def main():
    """
    Initializes and runs the GUI application.
    """
    root = tk.Tk()
    app = OrbitPropagatorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
