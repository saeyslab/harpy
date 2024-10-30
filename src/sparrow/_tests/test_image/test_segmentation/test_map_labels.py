from spatialdata import SpatialData

from sparrow.image import map_labels


def test_map_labels(sdata_multi_c: SpatialData):
    def _copy(img):
        return img

    depth = (100, 120)

    sdata_multi_c = map_labels(
        sdata_multi_c,
        func=_copy,
        labels_layers="masks_whole",
        output_labels_layer="masks_whole_copy",
        output_shapes_layer=None,
        depth=depth,
        overwrite=True,
        chunks=256,
        relabel_chunks=True,
    )
