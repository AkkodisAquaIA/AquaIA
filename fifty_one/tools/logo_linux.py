import sys
import os
from PIL import Image, ImageDraw, ImageTk, ImageFont
import tkinter as tk


# ============================================================================
def splash_screen_circle(image_path, duration=3000):
    """
    Splash screen circulaire avec un cercle extérieur clair
    et texte centré foncé.
    """                                                                          
    splash = tk.Tk()

    splash.overrideredirect(True)
    splash.attributes("-topmost", True)
    splash.attributes("-alpha", 0.0)

    # Couleur utilisée uniquement sous Windows
    transparent_color = "black"

    splash.configure(bg=transparent_color)

    # ------------------------------------------------------------------------
    # Chargement image
    # ------------------------------------------------------------------------
    img = Image.open(image_path).convert("RGBA")

    size = min(img.width, img.height)

    img = img.resize((size, size))

    # ------------------------------------------------------------------------
    # Création image circulaire
    # ------------------------------------------------------------------------
    img_circle = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    draw = ImageDraw.Draw(img_circle)

    # Couleurs
    circle_color = (64, 224, 208, 255)   # turquoise clair
    text_color = (0, 102, 102, 255)      # bleu-vert foncé

    # Cercle extérieur
    border_width = size // 20

    draw.ellipse(
        (0, 0, size, size),
        fill=circle_color
    )

    # ------------------------------------------------------------------------
    # Masque circulaire
    # ------------------------------------------------------------------------
    inner_size = size - 2 * border_width

    img_resized = img.resize((inner_size, inner_size))

    mask_inner = Image.new("L", (inner_size, inner_size), 0)

    draw_mask = ImageDraw.Draw(mask_inner)

    draw_mask.ellipse(
        (0, 0, inner_size, inner_size),
        fill=255
    )

    # Collage image centrée
    img_circle.paste(
        img_resized,
        (border_width, border_width),
        mask_inner
    )

    # ------------------------------------------------------------------------
    # Texte centré
    # ------------------------------------------------------------------------
    font_size = size // 8

    try:

        if sys.platform.startswith("win"):
            font_path = "arial.ttf"
        else:
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

        font = ImageFont.truetype(font_path, font_size)

    except:
        font = ImageFont.load_default()

    text = "Aqua-IA"

    draw_text = ImageDraw.Draw(img_circle)

    bbox = draw_text.textbbox((0, 0), text, font=font)

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    draw_text.text(
        ((size - w) / 2, (size - h) / 2),
        text,
        font=font,
        fill=text_color
    )

    # ------------------------------------------------------------------------
    # Conversion Tkinter
    # ------------------------------------------------------------------------
    photo = ImageTk.PhotoImage(img_circle)

    # IMPORTANT :
    # Canvas = évite le cadre cyan sous Linux
    canvas = tk.Canvas(
        splash,
        width=size,
        height=size,
        highlightthickness=0,
        bd=0,
        bg=transparent_color
    )

    canvas.pack()

    canvas.create_image(
        size // 2,
        size // 2,
        image=photo
    )

    # garder référence image
    canvas.image = photo

    # ------------------------------------------------------------------------
    # Transparence Windows uniquement
    # ------------------------------------------------------------------------
    if sys.platform.startswith("win"):
        splash.wm_attributes(
            "-transparentcolor",
            transparent_color
        )

    # ------------------------------------------------------------------------
    # Centrage écran
    # ------------------------------------------------------------------------
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()

    x = (screen_width - size) // 2
    y = (screen_height - size) // 2

    splash.geometry(f"{size}x{size}+{x}+{y}")

    # ------------------------------------------------------------------------
    # Fade in
    # ------------------------------------------------------------------------
    splash.after(
        10,
        lambda: fade_in(splash, steps=60, delay=20)
    )

    # ------------------------------------------------------------------------
    # Fade out
    # ------------------------------------------------------------------------
    splash.after(
        duration,
        lambda: fade_out(splash, steps=40, delay=20)
    )

    splash.mainloop()


# ============================================================================
def fade_in(window, steps=60, delay=20):

    alpha = 0.0

    increment = 1.0 / steps

    def _fade():

        nonlocal alpha

        alpha += increment

        if alpha >= 1.0:

            window.attributes("-alpha", 1.0)

        else:

            window.attributes("-alpha", alpha)

            window.after(delay, _fade)

    _fade()


# ============================================================================
def fade_out(window, steps=40, delay=20):

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


# ============================================================================
if __name__ == "__main__":

    splash_screen_circle(
        "mon_image.png",
        duration=4000
    )
    