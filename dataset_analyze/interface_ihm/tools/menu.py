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

# Local application imports
from config import constants as ct
from tools import utility as util
from tools import display_color as dc
from tools.display_color import DISPLAY_COLORS as colors

# Initialize colorama
init(autoreset=True)

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

#------------------------------------------------------------------------------

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
                 style: str = '') -> None:

        self.display = dc.DisplayColor()
        self.selected_menu = selected_menu
        self.dynamic_menu = dynamic_menu
        self.style = style

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
            print(f"{self.frame[1]}{self.frame[10] * box_width}{self.frame[3]}")
            print(f"{self.frame[11]}{self.board[0].center(box_width )}{self.frame[11]}")
            print(f"{self.frame[4]}{self.frame[10] * box_width}{self.frame[6]}")
            for i in range(1, self.ligne):
                print(f"{self.frame[11]} {i} : {self.board[i].ljust(max_length)} {self.frame[11]}")
            print(f"{self.frame[7]}{self.frame[10] * box_width}{self.frame[9]}")

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
