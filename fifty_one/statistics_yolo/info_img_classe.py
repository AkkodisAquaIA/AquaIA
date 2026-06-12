
import tools.display_color as dc
from config.constants import DISPLAY_COLORS as colors

#==============================================================================

#==============================================================================


def info_images_par_classe(data_info_img_cla, file):

    display = dc.DisplayColor()
 
    class_to_images = data_info_img_cla[0]
    class_names = data_info_img_cla[1]
 
    display.print("Nombres d'images par classe", colors['titre'])

    MAX_IMAGES_DISPLAY = 30     # nombre max d'images par classe
    MAX_CLASSES_SELECT = 6      # max classes que l'utilisateur peut demander

    def parse_selection(user_input, available_classes):
        """
        Parse une entrée du type:
        1 3-5 8
        Retourne une liste d'entiers uniques et valides.
        """
        result = set()
        parts = user_input.split()

        for part in parts:
            if "-" in part:
                try:
                    start, end = map(int, part.split("-"))
                    for i in range(start, end + 1):
                        if i in available_classes:
                            result.add(i)
                except ValueError:
                    continue
            else:
                try:
                    num = int(part)
                    if num in available_classes:
                        result.add(num)
                except ValueError:
                    continue

        return sorted(result)

    # available_classes = sorted(class_to_images.keys())
    available_classes = sorted(
        class_to_images.keys(),
        key=lambda cls: (-len(class_to_images[cls]), cls)
    )

    # Compute max lengths dynamically
    max_name_length = max(
        len(class_names[cls]) if class_names and cls < len(class_names) else len(f"UNK_{cls}")
        for cls in available_classes
    )

    max_count_length = max(
        len(str(len(class_to_images[cls]))) 
        for cls in available_classes
    )

    rows = []
    for cls in available_classes:
        name = class_names[cls] if class_names and cls < len(class_names) else f"UNK_{cls}"
        count = len(class_to_images[cls])
        formatted = f"{cls:>3} - {name:<{max_name_length}} : {count:>{max_count_length}}"
        rows.append(formatted)

    COLUMNS = 5
    col_width = max(len(r) for r in rows) + 0

    for i in range(0, len(rows), COLUMNS):
        line = rows[i:i + COLUMNS]
        print("│ ".join(f"{item:<{col_width}}" for item in line))
    print()

    if not file :
        # Affichage des noms des images par classe
        tag = f"Affichage des noms des images par classe"
        display.print(tag, colors['titre'])
        while True:
            tag = f"Entrez jusqu'à {MAX_CLASSES_SELECT} classes (ex: 1 3-5 8) ou 'Return' pour quitter : "
            display.print(tag, colors['input']) 
            user_input = input("  > ").strip()

            if user_input.lower() == '':
                break

            selected_classes = parse_selection(user_input, available_classes)            

            if not selected_classes:
                print()
                display.print("Aucune classe valide sélectionnée.\n", colors['warning'])
                continue

            if len(selected_classes) > MAX_CLASSES_SELECT:
                print()
                display.print(f"Maximum {MAX_CLASSES_SELECT} classes autorisées.\n", colors['warning'])
                continue    

            for cls in selected_classes:

                name = class_names[cls] if class_names and cls < len(class_names) else f"UNK_{cls}"
                all_images = sorted(class_to_images[cls])
                images = all_images[:MAX_IMAGES_DISPLAY]

                print(f"\n{cls:>2} {name}  ({len(all_images)} images)")

                if images:
                    max_width = max(len(img) for img in images) + 1
                    for i in range(0, len(images), 5):
                        ligne = images[i:i+5]
                        print(" │ ".join(f"{img:<{max_width}}" for img in ligne))
                        
                if len(all_images) > MAX_IMAGES_DISPLAY:
                    display.print(f"... + {len(all_images) - MAX_IMAGES_DISPLAY} autres images", colors['warning'])
            print()
