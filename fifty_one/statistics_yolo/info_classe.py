
from pathlib import Path
from colorama import Fore, Style, init
from collections import defaultdict

from tools import utility as util
from tools import system as syst
import tools.display_color as dc
from tools.display_color import DISPLAY_COLORS as colors
from config import constants as ct
from tools import graphe as gr

init(autoreset=True)

#================================================================================

#================================================================================

def info_classes(mode_aff, info_classes, cfg): 
  
    display = dc.DisplayColor()

    mode_affichage = mode_aff[0]
    etat_dataset = mode_aff[1]
    nom_dataset = mode_aff[2]


    while  True:

        if mode_affichage == ct.ECRAN:
            syst.clear_screen()

        display.print("(2) Information sur les classes", colors[etat_dataset], nom_dataset) # type: ignore
        class_distribution = info_classes[0]
        class_names =  info_classes[1]

        total = sum(class_distribution.values())

        items = sorted(class_distribution.items())

        max_name_len = max(
            len(class_names[cls] if class_names and cls < len(class_names) else f"UNK_{cls}")
            for cls, _ in items
        )

        blocs = []
        dom, moy, rary = 0, 0, 0

        # Tri par classes
        items_by_class = sorted(class_distribution.items())

        # Tri par fréquence
        items_by_freq = sorted(
            class_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )

        items = items_by_freq

        for cls, count in items :   #  items
            pct = (count / total) * 100
            name = class_names[cls] if class_names and cls < len(class_names) else f"UNK_{cls}"

            # Couleur automatique selon importance
            if pct < cfg["RARE"] :
                ccolor = Fore.RED
                rary +=1
            elif pct < cfg["DOMINANT"] :
                ccolor = Fore.YELLOW
                moy +=1
            else:
                ccolor = Fore.GREEN
                dom +=1

            BAR_WIDTH = 20 # largeur maximale de la barre pour 100% (ajustable)
            bbare = util.draw_bar(pct, 0, 100, BAR_WIDTH)
            ccount = util.format_nombre(count) # 1234 -> 1 234

            bloc = (
                f"{ccolor}"
                f"{cls:>2} "
                f"{name:<{max_name_len}} "
                f'{ccount:>8} '   
                f"{pct:5.2f}% "
                f"{bbare}"
                f"{Style.RESET_ALL}"
            )

            blocs.append(bloc)

        tag = f"Répartition des classes par fréquences ({rary + moy + dom})"
        display.print(tag , colors['titre'])
        legend_colored = (
            f'{Fore.GREEN}■ ({dom}) ≥ {cfg["DOMINANT"]}% Dominant{Style.RESET_ALL}   '
            f'│ {Fore.YELLOW}■ ({moy}) {cfg["RARE"]}–{cfg["DOMINANT"]}% Moyen{Style.RESET_ALL}   '
            f'│ {Fore.RED}■ ({rary}) < {cfg["RARE"]}% Rare{Style.RESET_ALL}'
        ).center(ct.DISPLAY_WIDTH)
        print(f"{legend_colored}\n")

        # Largeur terminal
        term_width = ct.DISPLAY_WIDTH # shutil.get_terminal_size().columns (317)
        bloc_width = max(len(b) for b in blocs) + 1
        classes_par_ligne = max(1, term_width // bloc_width)

        for i in range(0, len(blocs), classes_par_ligne):
            ligne = blocs[i:i + classes_par_ligne]
            print("│ ".join(f"{b:<{bloc_width}}" for b in ligne))


        print()
        # Affichage de l'histogramme de distribution des classes
        if mode_affichage == ct.ECRAN and ( util.answer_yes_or_no("Voulez-vous afficher le graphique") ) :
            display.print("Attente fermeture du graphe", colors['wait'])
            gr.histogram_classe(items, class_names, cfg, total )    

        if  mode_affichage == ct.FICHIER or not util.answer_yes_or_no("Voulez-vous modifier la valeur des seuils") : 
            break         

        cfg["RARE"], cfg["DOMINANT"]=  util.seuil() 


    # --- classes Rares ---------------------------------------
    if  mode_affichage == ct.ECRAN and ( util.answer_yes_or_no("Voulez-vous voir les classes rares") ):

        syst.clear_screen()
        display.print("(2) Information sur les classes rares", colors[etat_dataset], data) # type: ignore
        
        if cfg["RARE"] is not None:
            classes_faibles = []

            for cls, count in items:
                pct = (count / total) * 100
                if pct < cfg["RARE"]:
                    name = class_names[cls] if class_names and cls < len(class_names) else f"UNKNOWN_{cls}"
                    classes_faibles.append((cls, name, count, pct))

            if not classes_faibles:
                display.print(f'Aucune classe sous {cfg["RARE"]}% ', colors['ok'])
            else:
                message = f'Classes Rares ({rary}) < {cfg["RARE"]}%'
                display.print(message, colors['titre'])
                # tri optionnel (du pire au moins pire)
                classes_faibles.sort(key=lambda x: x[3])  # tri par %

                # --- regroupement par pourcentage ---
                grouped = defaultdict(list)

                for cls, name, count, pct in classes_faibles:
                    key = round(pct, 2)  # regroupe par % arrondi
                    grouped[key].append((cls, name, count))

                # tri par % croissant
                for pct in sorted(grouped.keys()):
                    print(f"--- {pct:.2f}% ---")

                    entries = grouped[pct]
                    texts = [f"{cls} {name}" for cls, name, _ in entries]

                    max_width = max(len(t) for t in texts) + 2
                    for i in range(0, len(texts), ct.N_PER_LINE):
                        print(" │ ".join(f"{t:<{max_width}}" for t in texts[i:i+ ct.N_PER_LINE]))
                    print()
                    