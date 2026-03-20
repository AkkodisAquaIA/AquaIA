import matplotlib.pyplot as plt
from collections import Counter


def bbox_overflow(file, overflow_warning, overflow_error):
    plt.figure(figsize=(12,6))
    counts, bins, patches = plt.hist(file, bins=50, color='skyblue', edgecolor='black')

    # Lignes verticales avec valeur dans la légende
    plt.axvline(overflow_warning, color='orange', linestyle='--', linewidth=3,
                label=f'Warning ({overflow_warning:.2f}%)')
    plt.axvline(overflow_error, color='red', linestyle='--', linewidth=3,
                label=f'Error ({overflow_error:.2f}%)')
    
    plt.text(overflow_warning, plt.ylim()[1]*0.95, f'{overflow_warning:.2f}%', 
             color='orange', fontsize=12, rotation=90, va='top', ha='right')
    plt.text(overflow_error, plt.ylim()[1]*0.95, f'{overflow_error:.2f}%', 
             color='red', fontsize=12, rotation=90, va='top', ha='right')

    plt.xlabel('Outside ratio (%)')
    plt.ylabel('Nombre de bbox')
    plt.title('Distribution des bbox hors limites')

    plt.legend()  # affichage de la légende
    plt.show()


def histograme(file, titre, x_label, y_label):
   
    plt.figure(figsize=(12,6))
    plt.hist(file, bins=50)
    plt.title(titre)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def histo_multipl(file, y_label, fault):

    plt.figure(figsize=(12,6))
    plt.bar(file.keys(), file.values(),
            color=[ 'red' if 'error' in t else 'orange' for t in file.keys() ])
    plt.xticks(rotation=45, ha='right')
    plt.title("Nombre d'anomalies par type")
    plt.ylabel(y_label)
    plt.show()

