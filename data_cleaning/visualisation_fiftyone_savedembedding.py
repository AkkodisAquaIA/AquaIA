import fiftyone as fo
import fiftyone.brain as fob

# Pour afficher les datasets enregistrés :
print(fo.list_datasets())
# Pour supprimer:
# fo.delete_dataset("dedup_dinov3_Diura_nan")
Saved_embedding_name = "coco-2017-validation"

if Saved_embedding_name in fo.list_datasets():
    dataset = fo.load_dataset(Saved_embedding_name)

    print("n total:", len(dataset))
    print("n avec dinov3_vec:", len(dataset.exists("dinov3_vec")))
    print("Brain runs:", dataset.list_brain_runs())

    view = dataset.exists("dinov3_vec")
    if len(view) < 2:
        raise ValueError("Pas assez d'échantillons avec embeddings pour UMAP")

    viz_key = "dinov3_umap"
    if viz_key in dataset.list_brain_runs():
        dataset.delete_brain_run(viz_key)
        dataset.save()

    fob.compute_visualization(
        view,
        embeddings="dinov3_vec",
        method="umap",
        brain_key=viz_key,
    )

    dataset.save()
    print("UMAP créé:", viz_key)

    session = fo.launch_app(dataset)
    session.wait()

else:
    print("The embedding name is not saved in the FiftyOne dataset list.")
