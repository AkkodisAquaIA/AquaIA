from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path


# ============================================================
# PARAMÈTRES
# ============================================================

TARGET_SIZE = 6336
DISPLAY_WIDTH = 1100
DISPLAY_HEIGHT = 800


# ============================================================
# RECHERCHE DU TIF CORRESPONDANT
# ============================================================


def find_tif_pair(jpg_path, tif_dir):
    """
    Recherche dans le dossier TIF un fichier ayant le même nom
    que le JPG, indépendamment de l'extension et de sa casse.
    """

    stem = jpg_path.stem.lower()

    for tif_path in tif_dir.iterdir():
        if tif_path.is_file() and tif_path.suffix.lower() in [".tif", ".tiff"] and tif_path.stem.lower() == stem:
            return tif_path

    return None


# ============================================================
# APPLICATION
# ============================================================


class CropApplication:
    def __init__(self, root, base_dir):
        self.root = root
        self.base_dir = Path(base_dir)

        # ----------------------------------------------------
        # Dossiers
        # ----------------------------------------------------

        self.jpg_dir = self.base_dir / "JPG"
        self.tif_dir = self.base_dir / "TIF"

        if not self.jpg_dir.exists():
            messagebox.showerror("Erreur", f"Dossier JPG introuvable :\n{self.jpg_dir}")
            root.destroy()
            return

        if not self.tif_dir.exists():
            messagebox.showerror("Erreur", f"Dossier TIF introuvable :\n{self.tif_dir}")
            root.destroy()
            return

        # ----------------------------------------------------
        # Dossiers de sortie
        # ----------------------------------------------------

        self.output_jpg_dir = self.base_dir / "JPG_cropped"
        self.output_tif_dir = self.base_dir / "TIF_cropped"

        self.output_jpg_dir.mkdir(exist_ok=True)
        self.output_tif_dir.mkdir(exist_ok=True)

        # ----------------------------------------------------
        # Liste des JPG
        # ----------------------------------------------------

        self.jpg_files = sorted([p for p in self.jpg_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg"]])

        if not self.jpg_files:
            messagebox.showerror("Erreur", f"Aucune image JPG trouvée dans :\n{self.jpg_dir}")
            root.destroy()
            return

        print(f"{len(self.jpg_files)} JPG trouvés.")

        # ----------------------------------------------------
        # Variables
        # ----------------------------------------------------

        self.index = 0

        self.jpg_image = None
        self.tif_image = None

        self.jpg_path = None
        self.tif_path = None

        self.tk_image = None

        self.scale = 1

        self.crop_x = 0
        self.crop_y = 0

        # ----------------------------------------------------
        # Interface
        # ----------------------------------------------------

        self.root.title("Découpage JPG + TIF")

        self.title_label = tk.Label(root, font=("Arial", 13, "bold"))
        self.title_label.pack(pady=5)

        self.canvas = tk.Canvas(root)
        self.canvas.pack(padx=10, pady=5)

        self.info_label = tk.Label(root, font=("Arial", 11), justify="center")
        self.info_label.pack(pady=5)

        # ----------------------------------------------------
        # Boutons
        # ----------------------------------------------------

        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        self.previous_button = tk.Button(button_frame, text="← Précédente", command=self.previous_image, font=("Arial", 11), padx=15, pady=7)

        self.previous_button.pack(side="left", padx=5)

        self.save_button = tk.Button(button_frame, text="Valider et suivante →", command=self.save_and_next, font=("Arial", 11, "bold"), padx=20, pady=7)

        self.save_button.pack(side="left", padx=5)

        # ----------------------------------------------------
        # Événements souris
        # ----------------------------------------------------

        self.canvas.bind("<Button-1>", self.move_crop)

        self.canvas.bind("<B1-Motion>", self.move_crop)

        # ----------------------------------------------------
        # Première image
        # ----------------------------------------------------

        self.load_image()

    # ========================================================
    # CHARGEMENT D'UNE IMAGE
    # ========================================================

    def load_image(self):
        if self.index >= len(self.jpg_files):
            messagebox.showinfo("Terminé", "Toutes les images ont été traitées.")

            self.root.destroy()
            return

        # ----------------------------------------------------
        # JPG
        # ----------------------------------------------------

        self.jpg_path = self.jpg_files[self.index]

        # ----------------------------------------------------
        # TIF correspondant
        # ----------------------------------------------------

        self.tif_path = find_tif_pair(self.jpg_path, self.tif_dir)

        if self.tif_path is None:
            messagebox.showwarning("TIF manquant", f"Aucun TIF trouvé pour :\n{self.jpg_path.name}\n\nCette image sera ignorée.")

            self.index += 1
            self.load_image()
            return

        # ----------------------------------------------------
        # Ouverture
        # ----------------------------------------------------

        self.jpg_image = Image.open(self.jpg_path)
        self.tif_image = Image.open(self.tif_path)

        jpg_size = self.jpg_image.size
        tif_size = self.tif_image.size

        print("\n--------------------------------")
        print(f"JPG : {self.jpg_path.name}")
        print(f"TIF : {self.tif_path.name}")
        print(f"Dimensions JPG : {jpg_size}")
        print(f"Dimensions TIF : {tif_size}")

        # ----------------------------------------------------
        # Vérification dimensions
        # ----------------------------------------------------

        if jpg_size != tif_size:
            messagebox.showerror("Dimensions différentes", f"{self.jpg_path.name}\nJPG : {jpg_size}\nTIF : {tif_size}")

            return

        width, height = jpg_size

        if width < TARGET_SIZE or height < TARGET_SIZE:
            messagebox.showerror("Image trop petite", f"{self.jpg_path.name}\n\nDimensions : {width} × {height}\nMinimum : {TARGET_SIZE} × {TARGET_SIZE}")

            return

        # ----------------------------------------------------
        # Crop centré par défaut
        # ----------------------------------------------------

        self.crop_x = (width - TARGET_SIZE) // 2
        self.crop_y = (height - TARGET_SIZE) // 2

        # ----------------------------------------------------
        # Calcul taille aperçu
        # ----------------------------------------------------

        self.scale = min(DISPLAY_WIDTH / width, DISPLAY_HEIGHT / height)

        display_width = int(width * self.scale)
        display_height = int(height * self.scale)

        preview = self.jpg_image.copy()

        preview.thumbnail((display_width, display_height))

        self.tk_image = ImageTk.PhotoImage(preview)

        # ----------------------------------------------------
        # Canvas
        # ----------------------------------------------------

        self.canvas.delete("all")

        self.canvas.config(width=display_width, height=display_height)

        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        # ----------------------------------------------------
        # Rectangle
        # ----------------------------------------------------

        self.rectangle = self.canvas.create_rectangle(
            self.crop_x * self.scale, self.crop_y * self.scale, (self.crop_x + TARGET_SIZE) * self.scale, (self.crop_y + TARGET_SIZE) * self.scale, outline="red", width=4
        )

        # ----------------------------------------------------
        # Texte
        # ----------------------------------------------------

        self.title_label.config(text=(f"Image {self.index + 1} / {len(self.jpg_files)} — {self.jpg_path.name}"))

        self.update_info()

    # ========================================================
    # DÉPLACEMENT DU CROP
    # ========================================================

    def move_crop(self, event):
        width, height = self.jpg_image.size

        # Position du centre du crop
        x = event.x / self.scale - TARGET_SIZE / 2
        y = event.y / self.scale - TARGET_SIZE / 2

        # Empêche de sortir de l'image
        x = max(0, min(x, width - TARGET_SIZE))

        y = max(0, min(y, height - TARGET_SIZE))

        self.crop_x = int(x)
        self.crop_y = int(y)

        # Mise à jour rectangle
        self.canvas.coords(self.rectangle, self.crop_x * self.scale, self.crop_y * self.scale, (self.crop_x + TARGET_SIZE) * self.scale, (self.crop_y + TARGET_SIZE) * self.scale)

        self.update_info()

    # ========================================================
    # INFORMATIONS
    # ========================================================

    def update_info(self):
        width, height = self.jpg_image.size

        left = self.crop_x
        right = width - (self.crop_x + TARGET_SIZE)

        top = self.crop_y
        bottom = height - (self.crop_y + TARGET_SIZE)

        self.info_label.config(
            text=(f"Crop final : {TARGET_SIZE} × {TARGET_SIZE} px\n\nGauche supprimée : {left} px     Droite supprimée : {right} px\nHaut supprimé : {top} px     Bas supprimé : {bottom} px")
        )

    # ========================================================
    # SAUVEGARDE
    # ========================================================

    def save_current(self):
        crop_box = (self.crop_x, self.crop_y, self.crop_x + TARGET_SIZE, self.crop_y + TARGET_SIZE)

        print(f"Crop utilisé : {crop_box}")

        # ----------------------------------------------------
        # JPG
        # ----------------------------------------------------

        jpg_crop = self.jpg_image.crop(crop_box)

        jpg_output = self.output_jpg_dir / self.jpg_path.name

        jpg_crop.save(jpg_output, quality=95)

        # ----------------------------------------------------
        # TIF
        # ----------------------------------------------------

        tif_crop = self.tif_image.crop(crop_box)

        tif_output = self.output_tif_dir / self.tif_path.name

        tif_crop.save(tif_output)

        print(f"JPG sauvegardé : {jpg_output}")
        print(f"TIF sauvegardé : {tif_output}")

    # ========================================================
    # VALIDER + IMAGE SUIVANTE
    # ========================================================

    def save_and_next(self):
        self.save_current()

        self.index += 1

        self.load_image()

    # ========================================================
    # IMAGE PRÉCÉDENTE
    # ========================================================

    def previous_image(self):
        if self.index > 0:
            self.index -= 1
            self.load_image()


# ============================================================
# SÉLECTION DU DOSSIER PRINCIPAL
# ============================================================


def main():
    selection_root = tk.Tk()
    selection_root.withdraw()

    base_dir = filedialog.askdirectory(title=("Sélectionner le dossier contenant les dossiers JPG et TIF"))

    selection_root.destroy()

    if not base_dir:
        print("Aucun dossier sélectionné.")
        return

    print(f"Dossier sélectionné : {base_dir}")

    # --------------------------------------------------------
    # Application
    # --------------------------------------------------------

    root = tk.Tk()

    root.mainloop()


# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":
    main()
