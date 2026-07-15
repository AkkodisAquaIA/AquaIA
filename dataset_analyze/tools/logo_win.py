import sys
import os
from PIL import Image, ImageDraw, ImageTk, ImageFont
import tkinter as tk

#============================================================================
def splash_screen_circle(image_path, duration=3000):
    """
    Splash screen circulaire avec un cercle extérieur clair
    et texte centré foncé.
    """
    splash = tk.Tk()
    splash.overrideredirect(True)
    splash.attributes("-topmost", True)

    splash.attributes("-alpha", 0.0)

    transparent_color = "magenta"
    splash.configure(bg=transparent_color)

    # --- Charger l'image ---
    img = Image.open(image_path).convert("RGBA")
    size = min(img.width, img.height)
    img = img.resize((size, size))

    # --- Créer image finale ---
    img_circle = Image.new("RGBA", (size, size), (0,0,0,0))
    draw = ImageDraw.Draw(img_circle)

    # Couleurs
    circle_color = (64, 224, 208, 255)  # Turquoise clair
    text_color = (0, 102, 102, 255)     # Bleu-vert foncé

    # Dessiner le cercle extérieur
    border_width = size // 20
    draw.ellipse((0, 0, size, size), fill=circle_color)

    # Masque circulaire pour l'image
    inner_size = size - 2*border_width
    img_resized = img.resize((inner_size, inner_size))
    mask_inner = Image.new("L", (inner_size, inner_size), 0)
    draw_mask_inner = ImageDraw.Draw(mask_inner)
    draw_mask_inner.ellipse((0,0,inner_size, inner_size), fill=255)

    # Coller l'image centrée
    img_circle.paste(img_resized, (border_width, border_width), mask_inner)

    try:
        font_path = "arial.ttf"
        if not os.path.exists(font_path):
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = ImageFont.truetype(font_path, font_size) # type: ignore
    except:
        font = ImageFont.load_default()


    # Ajouter le texte centré
    draw_text = ImageDraw.Draw(img_circle)
    font_size = size // 8
    try:
        font = ImageFont.truetype("arial.ttf", font_size) # type: ignore
    except:
        font = ImageFont.load_default() # type: ignore
    text = "Aqua-IA"
    bbox = draw_text.textbbox((0,0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw_text.text(((size-w)/2, (size-h)/2), text, font=font, fill=text_color)

    # # Conversion en image Tkinter
    # photo = ImageTk.PhotoImage(img_circle)
    # label = tk.Label(splash, image=photo, bg=transparent_color, bd=0)
    # label.pack()

    photo = ImageTk.PhotoImage(img_circle)
    label = tk.Label(splash, image=photo, bg=transparent_color, bd=0)
    label.image = photo  # type: ignore # <-- IMPORTANT
    label.pack()

    if sys.platform.startswith("win"):
        splash.wm_attributes("-transparentcolor", transparent_color)

    # Centrer la fenêtre
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    x = (screen_width - size) // 2
    y = (screen_height - size) // 2
    splash.geometry(f"{size}x{size}+{x}+{y}")

    splash.after(10, lambda: fade_in(splash, steps=60, delay=30))
    
    # Afficher le splash screen pour la durée
    splash.after(duration, lambda: fade_out(splash, steps=40))
    splash.mainloop()

def fade_in(window, steps=60, delay=40):
    alpha = 0.0
    increment = 1.0 / steps

    def _fade():
        nonlocal alpha
        alpha += increment
        if alpha >= 1:
            window.attributes("-alpha", 1.0)
        else:
            window.attributes("-alpha", alpha)
            window.after(delay, _fade)

    _fade()


def fade_out(window, steps=60, delay=40):
    alpha = 1.0
    decrement = 1.0 / steps

    def _fade():
        nonlocal alpha
        alpha -= decrement
        if alpha <= 0:
            window.destroy()
        else:
            window.attributes("-alpha", alpha)
            window.after(delay, _fade)

    _fade()
