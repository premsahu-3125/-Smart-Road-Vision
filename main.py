"""
main.py

Entry point. Kept intentionally thin - all real logic lives in gui.py,
detector.py, logger.py, and report.py.
"""

import tkinter as tk

from gui import SmartRoadVisionApp


def main():
    root = tk.Tk()
    app = SmartRoadVisionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_exit)
    root.mainloop()


if __name__ == "__main__":
    main()
