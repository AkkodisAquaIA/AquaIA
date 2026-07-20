# -*- coding: utf-8 -*-
"""
This module allows creating a menu and managing the input choice based on the selection.
You can choose the frame format of the menu.
It is possible to opt for a static menu or a dynamic menu.

@author: Pierre.FANCELLI
"""
# Standard library
import sys
from typing import Any, Dict, List, Tuple

# Third-party
from colorama import init, Style
from dataclasses import dataclass

# Local application imports
from config import constants as ct
from tools import utility as util
from tools import display_color as dc
from tools.display_color import DISPLAY_COLORS as colors

# Initialize colorama
init(autoreset=True)

#============================================================================
@dataclass(frozen=True)
class MenuTheme:
    frame: tuple
    title: tuple
    number: tuple
    text: tuple


DEFAULT_THEME = MenuTheme(
    frame=  (250,250,250,""),
    title=  (255,153, 51,""),
    number= (255, 16,240,""),
    text=   (204,204,  0,"")
)


AQUA_IA = MenuTheme(
    frame=  (255, 255, 255,""), # Blanc
    title=  (102, 255, 204,""), # Light turquoise   
    number= ( 51, 102, 255,""), #  Bleu
    text=   (200,200, 200,"")   # Blanc 
)

DARK_THEME = MenuTheme(
    frame=  (100,100,100,""),
    title=  (100,100,100,""),
    number= (100,100,100,""),
    text=   (100,100,100,"")
)


#============================================================================


# symbols for frame creation
PATTERN ={
"double" : [".","╔", "╦", "╗",
                "╠", "╬", "╣",
                "╚", "╩", "╝",
            "═", "║"
           ],
"simple" : [".","┌", "┬", "┐",
                "├", "┼", "┤",
                "└", "┴", "┘",
            "─", "│"
           ],
"rounds" : [".","╭", "┬", "╮",
                "├", "┼", "┤",
                "╰", "┴", "╯",
            "─", "│"
           ],
"heavy": [".","┏", "┳", "┓",   
              "┣", "╋", "┫",   
              "┗", "┻", "┛",   
        "━", "┃"         
        ],           
"Unicode" : [".","┏", "┯", "┓",
                 "┣", "┿", "┫",
                 "┗", "┷", "┛",
        "━", "┃","│"
        ],
"ASCII" : ["." ,"+" , "+" , "+",
               "+", "+", "+",
               "+", "+", "+",
           "─", "│"
           ],
    }

#----------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------

# Menu class
class Menu :
    """
    This class allows creating a menu and managing the input choice based on the selection.
    You can choose the frame format of the menu.

    You can choose the style of the menu frame between:
    'simple', 'double', 'heavy', 'rounds', 'ASCII', or 'Unicode'.
    The default style is 'double'.
    """


    def __init__(self, selected_menu: str,
                 dynamic_menu: List[str] | None = None,
                 style: str = '',
                 theme: MenuTheme = DEFAULT_THEME) -> None:

        self.display = dc.DisplayColor()
        self.selected_menu = selected_menu
        self.dynamic_menu = dynamic_menu
        self.style = style
        self.theme = theme

        # Check selec menu
        self.unknown_menu = False
        if self.selected_menu == 'Dynamic':
            if self.dynamic_menu is None :
                text = f" '{self.selected_menu}' : Menu manquant."
                self.display.print(text, colors['error'])
                sys.exit()
            else:
                self.board = [str(element) for element in self.dynamic_menu]
        elif self.selected_menu in ct.MENUS:
            self.board = [str(element) for element in ct.MENUS[self.selected_menu]]
        else:
            self.unknown_menu = True

        # Select pattern
        style_mapping = {
            'simple': PATTERN['simple'],
            'double' : PATTERN['double'],
            'rounds': PATTERN['rounds'],
            'heavy': PATTERN['heavy'],
            'ASCII' : PATTERN['ASCII'],
            'Unicode' : PATTERN['Unicode']
        }
        self.frame = style_mapping.get(self.style, PATTERN['double'])

        self.ligne = 0

    
    def color_frame(self, text: str) -> str:
        """
        Apply the frame color defined in the current theme.
        """
        return self.display.colored(
            text,
            self.theme.frame,
            pref=False
        )

    def display_menu(self) -> None:
        """        
        Method to display the menu based on the selected style and content.
        """

        if self.unknown_menu :

            text = f"'{self.selected_menu}' : Ce menu n'existe pas !"
            self.display.print(text, colors['error'])
            sys.exit()
        else:
            max_length = max(len(chaine) for chaine in self.board)
            self.ligne = len(self.board)
            box_width = max_length + 6

            # Create the pattern
            # Top of the Frame            
            top = (
                f"{self.frame[1]}"
                f"{self.frame[10] * box_width}"
                f"{self.frame[3]}"
            )

            print(self.color_frame(top))

                        
            # Titre du menu
            left_right = self.color_frame(self.frame[11])

            title = self.display.colored(
                self.board[0].center(box_width),
                self.theme.title,         
                pref=False,
                bold=True
            )

            print(f"{left_right}{title}{left_right}")

            
            # Affichage de la ligne de séparation entre le titre et les choix du menu
            separator = (
                f"{self.frame[4]}"
                f"{self.frame[10] * box_width}"
                f"{self.frame[6]}"
            )

            print(self.color_frame(separator))


            # Affichage des choix du menu
            for i in range(1, self.ligne):
            
                number = self.display.colored(f"{i}", self.theme.number, pref=False, bold=True)

                text = self.display.colored(
                    self.board[i].ljust(max_length),
                    self.theme.text,
                    pref=False,
                    bold=True
                )

                side = self.color_frame(self.frame[11])
                print(f"{side} {number} : {text} {side}")


            # Bottom of the Frame
            bottom = (
                f"{self.frame[7]}"
                f"{self.frame[10] * box_width}"
                f"{self.frame[9]}"
            )

            print(self.color_frame(bottom))


            if self.ligne < 2 :
                text = f" '{self.board[0]}' : Menu sans choix !"
                self.display.print(text, colors['error'])
                sys.exit()


    def selection(self) -> int:
        """
        Method to manage the user's choice in the menu.
        """
        color = colors['input']

        while True:
            try:
                #  # Convert the input color from DISPLAY_COLORS to ANSI
                input_color = util.rgb_to_ansi(color[:3])
                # # Displays the prompt in color
                prompt = "Faites votre choix"
                colored_select = f"{input_color}[?] {prompt}: {Style.RESET_ALL}"

                select = int(input(colored_select).strip())

                if 1 <= select <= self.ligne - 1 :
                    return select
                text = f"Oops! la sélection n'est pas valide. Essayez encore ! {ct.BELL}"
                self.display.print(text, colors['error'])

            # Input is not a number
            except ValueError:
                text = (
                        f"Il semble que vous n'ayez pas saisi un numéro valide. "
                        f"Essayez encore! {ct.BELL}"
                    )
                self.display.print(text, colors['error'])
