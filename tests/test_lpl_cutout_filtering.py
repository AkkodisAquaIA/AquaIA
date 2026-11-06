from data_processing.crop_detection_to_classification import decode_class_name
from data_processing.filter_sam3_cutouts_with_lpl_annotations import match_detections


def test_decode_class_name_expands_known_ranks_even_when_order_is_unknown():
    prefixes = [
        {"Arthr": "Arthropoda"},
        {"Insec": "Insecta"},
        {"Coleo": "Coleoptera"},
    ]

    decoded, status = decode_class_name(
        "Arthr_Insec_Plectoptera_Leuctridae_Leuctra",
        prefixes,
    )

    assert decoded == "Arthropoda_Insecta_Plectoptera_Leuctridae_Leuctra"
    assert status == "converti_par_prefixes"


def test_match_detections_rejects_box_unrelated_to_annotations():
    detections = [
        (0, 0.50, 0.50, 0.20, 0.20),
        (0, 0.90, 0.90, 0.05, 0.05),
    ]
    annotations = [(42, 0.50, 0.50, 0.22, 0.22)]

    matches, used_annotations = match_detections(
        detections,
        annotations,
        min_detection_coverage=0.5,
        min_iou=0.1,
    )

    assert set(matches) == {0}
    assert matches[0]["annotation_index"] == 0
    assert used_annotations == {0}


def test_match_detections_assigns_an_annotation_only_once():
    detections = [
        (0, 0.50, 0.50, 0.20, 0.20),
        (0, 0.50, 0.50, 0.18, 0.18),
    ]
    annotations = [(7, 0.50, 0.50, 0.22, 0.22)]

    matches, used_annotations = match_detections(
        detections,
        annotations,
        min_detection_coverage=0.5,
        min_iou=0.1,
    )

    assert len(matches) == 1
    assert used_annotations == {0}
