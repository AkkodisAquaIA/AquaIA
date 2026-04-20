#!/usr/bin/env python3

import os
import argparse
import rawpy
import imageio.v3 as iio


def convert_cr3_to_tiff(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".cr3"):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + ".tif"
            output_path = os.path.join(output_dir, output_filename)

            print(f"Conversion : {filename} → {output_filename}")

            try:
                with rawpy.imread(input_path) as raw:

                    rgb = raw.postprocess(
                        # Balance des blancs robuste
                        use_camera_wb=False,   # souvent foireux sur CR3
                        use_auto_wb=True,      # beaucoup plus fiable

                        # Luminosité correcte
                        no_auto_bright=False,
                        bright=1.2,            # ajuste si trop sombre

                        # Qualité maximale
                        output_bps=16,         # TIFF 16 bits
                        demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,

                        # Pas de réduction de taille
                        half_size=False
                    )

                iio.imwrite(output_path, rgb)

            except Exception as e:
                print(f"Erreur avec {filename} : {e}")

    print("Conversion terminée !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convertir des fichiers CR3 en TIFF")
    parser.add_argument("input_dir", help="Dossier contenant les fichiers CR3")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Dossier de sortie (optionnel)"
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or os.path.join(input_dir, "tiff_output")

    convert_cr3_to_tiff(input_dir, output_dir)