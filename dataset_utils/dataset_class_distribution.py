import os
import matplotlib.pyplot as plt

CLASSES_IA = [
"Polycentropodidae_Cyrnus_flavidus",
"Polycentropodidae_Plectrocnemia_sp",
"Phryganeidae_Agrypnia_sp",
"Lepidostomatidae_Lepidostoma_hirtum",
"Polycentropodidae_Neureclipsis_bimaculata",
"Hydroptilidae_Oxyethira_sp",
"Philopotamidae_Chimarra_sp",
"Leptoceridae_Ceraclea_sp",
"Goeridae_Silo_pallipes",
"Hydroptilidae_Ithytrichia_sp",
"Philopotamidae_Philopotamus_sp",
"Limnephilidae_Genus_sp",
"Heptageniidae_Heptagenia_dalecarlica",
"Heptageniidae_Heptagenia_sulphurea",
"Ephemerellidae_Ephemerella_aurivillii",
"Ephemerellidae_Ephemerella_mucronata",
"Leptophlebiidae_Habrophlebia_sp",
"Baetidae_Baetis_rhodani",
"Baetidae_Baetis_digitatus",
"Baetidae_Baetis_niger",
"Baetidae_Baetis_vernus",
"Ameletidae_Ameletus_inopinatus",
"Baetidae_Centroptilum_luteolum",
"Corixidae_Callicorixa_wollastoni",
"Planorbidae_Gyraulus_sp",
"Asellidae_Asellus_aquaticus",
"Sialidae_Sialis_lutaria",
"Sialidae_Sialis_fuliginosa",
"Sialidae_Sialis_morio",
"Chloroperlidae_Siphonoperla_burmeisteri",
"Perlodidae_Diura_nanseni",
"Perlodidae_Diura_others",
"Perlodidae_Diura_bicaudata",
"Taeniopterygidae_Brachyptera_risi",
"Nemouridae_Nemoura_sp",
"Nemouridae_Nemoura_avicularis",
"Nemouridae_Nemoura_cinerea",
"Nemouridae_Nemoura_flexuosa",
"Perlodidae_Isoperla_sp",
"Psychodidae_Genus_sp",
"Taeniopterygidae_Taeniopteryx_nebulosa",
"Nemouridae_Protonemura_sp",
"Leuctridae_Leuctra_sp",
"Leuctridae_Leuctra_fusca",
"Leuctridae_Leuctra_hippopus",
"Leuctridae_Leuctra_nigra",
"Sphaeriidae_Pisidium_sp",
"Pediciidae_Dicranota_sp",
"Empididae_Chelifera_sp",
"Empididae_Hemerodromia_sp",
"Athericidae_Atherix_sp",
"Chironomidae_Genus_sp",
"Ceratopogonidae_Genus_sp",
"Elmidae_Elmis_aenea_adult",
"Elmidae_Elmis_aenea_larva",
"Elmidae_Elmis_aenea",
"Scirtidae_Elodes_sp",
"Limoniidae_Eloeophila_sp",
"Gammaridae_Gammarus_sp",
"Hydraenidae_Hydraena_sp",
"Hydracarina_Genus_sp",
"Hydropsychidae_Hydropsyche_saxonica",
"Hydropsychidae_Hydropsyche_pellucidula",
"Hydropsychidae_Hydropsyche_siltalai",
"Leptophlebiidae_Paraleptophlebia_sp",
"Leptophlebiidae_Leptophlebia_sp",
"Elmidae_Limnius_volckmari",
"Brachycentridae_Micrasema_gelidum",
"Brachycentridae_Micrasema_setiferum",
"Elmidae_Oulimnius_tuberculatus_adult",
"Elmidae_Oulimnius_tuberculatus_larva",
"Elmidae_Oulimnius_tuberculatus",
"Polycentropodidae_Polycentropus_flavomaculatus",
"Polycentropodidae_Polycentropus_irroratus",
"Rhyacophilidae_Rhyacophila_fasciata",
"Rhyacophilidae_Rhyacophila_nubila",
"Simuliidae_Genus_sp",
"Sericostomatidae_Sericostoma_personatum",
"Glossosomatidae_Agapetus_sp",
"Nemouridae_Amphinemura_borealis",
"Capniidae_Capnopsis_schilleri",
"Heptageniidae_Kageronia_fuscogrisea",
"Sphaeriidae_Sphaerium_sp"
]
import os
import matplotlib.pyplot as plt

def plot_images_per_class(root_path, extensions=(".jpg", ".jpeg", ".png", ".tif", ".tiff")):
    """
    Histogramme du nombre d'images par classe.
    Si le dossier n'existe pas → 0 image.
    """

    counts = []
    c = 0

    for cls in CLASSES_IA:
        folder_path = os.path.join(root_path, cls)

        if os.path.isdir(folder_path):
            n_images = sum(
                1 for f in os.listdir(folder_path)
                if f.lower().endswith(extensions)
            )
            c += 1
        else:
            n_images = 0

        counts.append(n_images)

    print('nb class =', c)

    plt.figure(figsize=(16,6))

    bars = plt.bar(CLASSES_IA, counts)

    plt.xticks(rotation=90)
    plt.xlabel("Classes IA")
    plt.ylabel("Nombre d'images")
    plt.title("Nombre d'images par classe")

    # --- ajout du nombre d'images sur chaque barre ---
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            str(count),
            ha='center',
            va='bottom',
            fontsize=8
        )

    plt.tight_layout()
    plt.show()


plot_images_per_class("/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/AQUA-IA_dataset_mars2026_splited/train")
