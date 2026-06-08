import os
from tkinter import Tk, filedialog

def build_tree(path, file, prefix=""):
    entries = sorted(os.listdir(path))

    for i, entry in enumerate(entries):
        full_path = os.path.join(path, entry)

        connector = "└── " if i == len(entries) - 1 else "├── "
        line = prefix + connector + entry

        file.write(line + "\n")

        if os.path.isdir(full_path):
            extension = "    " if i == len(entries) - 1 else "│   "
            build_tree(full_path, file, prefix + extension)

root = Tk()
root.withdraw()

folder = filedialog.askdirectory(title="Оберіть папку")

if folder:
    with open("tree.txt", "w", encoding="utf-8") as f:
        f.write(os.path.basename(folder) + "\n")
        build_tree(folder, f)

    print("Готово. Результат збережено у tree.txt")