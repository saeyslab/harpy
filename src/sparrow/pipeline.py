import glob
import os
import warnings

from omegaconf import DictConfig, ListConfig
from spatialdata import SpatialData, read_zarr

import sparrow as sp
from sparrow.utils._keys import _CELLSIZE_KEY

log = sp.utils.get_pylogger(__name__)


class SparrowPipeline:
    """Sparrow pipeline."""

    def __init__(self, cfg: DictConfig, image_name: str = "raw_image"):
        self.cfg = cfg
        self.loaded_image_name = image_name
        self.cleaned_image_name = self.loaded_image_name
        self.shapes_layer_name = self.cfg.segmentation.output_shapes_layer

    def run_pipeline(self) -> SpatialData:
        """Run the pipeline."""
        # Checks the config paths, see the src/sparrow/configs and local configs folder for settings
        _check_config(self.cfg)

        # Supress _core_genes futerewarnings
        warnings.simplefilter(action="ignore", category=FutureWarning)

        # The actual pipeline which consists of 6 steps:

        # Load
        log.info("Converting to zarr and (lazy) loading of SpatialData object.")
        sdata = self.load()
        log.info("Conversion to zarr finished.")

        # Clean
        log.info("Cleaning step started.")
        sdata = self.clean(sdata)
        log.info("Cleaning step finished.")

        # Segment
        log.info("Segmentation step started.")
        sdata = self.segment(sdata)
        log.info("Segmentation step finished.")

        # Allocate
        log.info("Allocation step started.")
        sdata = self.allocate(sdata)
        log.info("Allocation step finished.")

        if self.cfg.dataset.markers is not None:
            # Annotate
            log.info("Annotation step started.")
            sdata = self.annotate(sdata)
            log.info("Annotation step finished.")

            # Visualize
            log.info("Visualization step started.")
            sdata = self.visualize(sdata)
            log.info("Visualization step finished.")

        return sdata

    def load(self) -> SpatialData:
        """Loading step, the first step of the pipeline, performs creation of spatial data object."""
        # cast to list if cfg.dataset.image is a ListConfig object (i.e. for multiple channels)
        if isinstance(self.cfg.dataset.image, ListConfig):
            filename_pattern = list(self.cfg.dataset.image)
        else:
            filename_pattern = str(self.cfg.dataset.image)

        if not isinstance(filename_pattern, list) and filename_pattern.endswith(".zarr"):
            sdata = read_zarr(filename_pattern)
            if isinstance(sdata, SpatialData):
                img_layers = [*sdata.images]
                if self.loaded_image_name not in img_layers:
                    raise ValueError(
                        f"Provided image layer '{self.loaded_image_name}' not in SpatialData object loaded from zarr."
                    )
                log.info(
                    f"Applying SparrowPipeline on '{self.loaded_image_name}' image layer in provided SpatialData object."
                )
                if self.cfg.dataset.image != self.cfg.paths.sdata:
                    # changing backing directory
                    sdata.write(self.cfg.paths.sdata)
                return sdata
            else:
                raise ValueError("Currently only zarr's of type SpatialData are supported.")

        else:
            log.info("Creating sdata.")
            sdata = sp.io.create_sdata(
                input=filename_pattern,
                output_path=self.cfg.paths.sdata,
                img_layer=self.loaded_image_name,
                crd=self.cfg.dataset.crop_param,
                chunks=self.cfg.dataset.chunks,
                scale_factors=self.cfg.dataset.scale_factors,
                z_projection=self.cfg.dataset.z_projection,
            )
            log.info("Finished creating sdata.")

        return sdata

    def clean(self, sdata: SpatialData) -> SpatialData:
        """Cleaning step, the second step of the pipeline, performs tilingCorrection and preprocessing of the image to improve image quality."""
        sp.pl.plot_image(
            sdata=sdata,
            output=os.path.join(self.cfg.paths.output_dir, "original"),
            crd=self.cfg.clean.small_size_vis,
            img_layer=self.loaded_image_name,
        )

        self.cleaned_image_name = self.loaded_image_name

        # Perform tilingCorrection on the whole image, corrects illumination and performs inpainting
        if self.cfg.clean.tilingCorrection:
            log.info("Start tiling correction.")

            output_layer = self.cfg.clean.output_img_layer_tiling_correction

            sdata, flatfields = sp.im.tiling_correction(
                sdata=sdata,
                img_layer=self.cleaned_image_name,
                crd=self.cfg.clean.crop_param if self.cfg.clean.crop_param is not None else None,
                scale_factors=self.cfg.dataset.scale_factors,
                tile_size=self.cfg.clean.tile_size,
                output_layer=output_layer,
                overwrite=self.cfg.clean.overwrite,
            )

            self.cleaned_image_name = output_layer

            log.info("Tiling correction finished.")

            # Write plot to given path if output is enabled
            if "tiling_correction" in self.cfg.paths:
                log.info(f"Writing tiling correction plot to {self.cfg.paths.tiling_correction}")
                sp.pl.tiling_correction(
                    sdata=sdata,
                    img_layer=[self.loaded_image_name, self.cleaned_image_name],
                    crd=self.cfg.clean.small_size_vis if self.cfg.clean.small_size_vis is not None else None,
                    output=self.cfg.paths.tiling_correction,
                )
                for i, flatfield in enumerate(flatfields):
                    # flatfield can be None is tiling correction failed.
                    if flatfield is not None:
                        sp.pl.flatfield(
                            flatfield,
                            output=f"{self.cfg.paths.tiling_correction}_flatfield_{i}",
                        )

            sp.pl.plot_image(
                sdata=sdata,
                output=os.path.join(self.cfg.paths.output_dir, self.cleaned_image_name),
                crd=self.cfg.clean.small_size_vis,
                img_layer=self.cleaned_image_name,
            )

        # min max filtering

        if self.cfg.clean.minmaxFiltering:
            log.info("Start min max filtering.")

            output_layer = self.cfg.clean.output_img_layer_min_max_filtering

            sdata = sp.im.min_max_filtering(
                sdata=sdata,
                img_layer=self.cleaned_image_name,
                crd=self.cfg.clean.crop_param if self.cfg.clean.crop_param is not None else None,
                size_min_max_filter=list(self.cfg.clean.size_min_max_filter)
                if isinstance(self.cfg.clean.size_min_max_filter, ListConfig)
                else self.cfg.clean.size_min_max_filter,
                scale_factors=self.cfg.dataset.scale_factors,
                output_layer=output_layer,
                overwrite=self.cfg.clean.overwrite,
            )

            self.cleaned_image_name = output_layer

            log.info("Min max filtering finished.")

            sp.pl.plot_image(
                sdata=sdata,
                output=os.path.join(self.cfg.paths.output_dir, self.cleaned_image_name),
                crd=self.cfg.clean.small_size_vis,
                img_layer=self.cleaned_image_name,
            )

        # contrast enhancement

        if self.cfg.clean.contrastEnhancing:
            log.info("Start contrast enhancing.")

            output_layer = self.cfg.clean.output_img_layer_clahe

            sdata = sp.im.enhance_contrast(
                sdata=sdata,
                img_layer=self.cleaned_image_name,
                crd=self.cfg.clean.crop_param if self.cfg.clean.crop_param is not None else None,
                contrast_clip=list(self.cfg.clean.contrast_clip)
                if isinstance(self.cfg.clean.contrast_clip, ListConfig)
                else self.cfg.clean.contrast_clip,
                chunks=self.cfg.clean.chunksize_clahe,
                depth=self.cfg.clean.depth,
                output_layer=output_layer,
                scale_factors=self.cfg.dataset.scale_factors,
                overwrite=self.cfg.clean.overwrite,
            )

            self.cleaned_image_name = output_layer

            log.info("Contrast enhancing finished.")

            sp.pl.plot_image(
                sdata=sdata,
                output=os.path.join(self.cfg.paths.output_dir, self.cleaned_image_name),
                crd=self.cfg.clean.small_size_vis,
                img_layer=self.cleaned_image_name,
            )

        return sdata

    def segment(self, sdata: SpatialData) -> SpatialData:
        """Segmentation step, the third step of the pipeline, performs cellpose segmentation and creates masks."""
        log.info("Start segmentation.")

        depth = self.cfg.segmentation.depth
        if isinstance(depth, ListConfig):
            depth = tuple(depth)
        elif depth is None:
            log.info(
                f"Depth not provided for segmentation, "
                f"setting depth equal to 2 times the estimated size of the nucleus/cell: 2*{ self.cfg.segmentation.diameter}"
            )
            depth = 2 * self.cfg.segmentation.diameter

        self.shapes_layer_name = self.cfg.segmentation.output_shapes_layer
        self.labels_layer_name = self.cfg.segmentation.output_labels_layer

        # Perform segmentation
        sdata = sp.im.segment(
            sdata=sdata,
            img_layer=self.cleaned_image_name,
            output_labels_layer=self.labels_layer_name,
            output_shapes_layer=self.shapes_layer_name,
            depth=depth,
            chunks=self.cfg.segmentation.chunks,
            trim=self.cfg.segmentation.trim,
            crd=self.cfg.segmentation.crop_param if self.cfg.segmentation.crop_param is not None else None,
            scale_factors=self.cfg.dataset.scale_factors,
            device=self.cfg.device,
            min_size=self.cfg.segmentation.min_size,
            flow_threshold=self.cfg.segmentation.flow_threshold,
            diameter=self.cfg.segmentation.diameter,
            cellprob_threshold=self.cfg.segmentation.cellprob_threshold,
            model_type=self.cfg.segmentation.model_type,
            channels=list(self.cfg.segmentation.channels)
            if isinstance(self.cfg.segmentation.channels, ListConfig)
            else self.cfg.segmentation.channels,
            do_3D=self.cfg.segmentation.do_3D,
            anisotropy=self.cfg.segmentation.anisotropy,
            overwrite=self.cfg.segmentation.overwrite,
        )

        log.info("Segmentation finished.")

        sp.pl.segment(
            sdata=sdata,
            crd=self.cfg.segmentation.small_size_vis
            if self.cfg.segmentation.small_size_vis is not None
            else self.cfg.clean.small_size_vis,
            img_layer=self.cleaned_image_name,
            shapes_layer=self.shapes_layer_name,
            output=self.cfg.paths.segmentation,
        )

        if self.cfg.segmentation.expand_radius:
            sdata = sp.im.expand_labels_layer(
                sdata,
                labels_layer=self.labels_layer_name,
                distance=self.cfg.segmentation.expand_radius,
                chunks=self.cfg.segmentation.chunks,
                output_labels_layer=f"expanded_cells_labels_{self.cfg.segmentation.expand_radius}",
                output_shapes_layer=f"expanded_cells_shapes_{self.cfg.segmentation.expand_radius}",
                overwrite=True,
            )

            # update current shapes layer name
            self.shapes_layer_name = f"expanded_cells_shapes_{self.cfg.segmentation.expand_radius}"
            self.labels_layer_name = f"expanded_cells_labels_{self.cfg.segmentation.expand_radius}"

            sp.pl.segment(
                sdata=sdata,
                crd=self.cfg.segmentation.small_size_vis
                if self.cfg.segmentation.small_size_vis is not None
                else self.cfg.clean.small_size_vis,
                img_layer=self.cleaned_image_name,
                shapes_layer=self.shapes_layer_name,
                output=f"{self.cfg.paths.segmentation}_expanded_cells_{self.cfg.segmentation.expand_radius}",
            )

        return sdata

    def allocate(self, sdata: SpatialData) -> SpatialData:
        """Allocation step, the fourth step of the pipeline, creates the adata object from the mask and allocates the transcripts from the supplied file."""
        sdata = sp.io.read_transcripts(
            sdata,
            path_count_matrix=self.cfg.dataset.coords,
            transform_matrix=self.cfg.dataset.transform_matrix,
            output_layer=self.cfg.allocate.points_layer_name,
            overwrite=self.cfg.allocate.overwrite,
            delimiter=self.cfg.allocate.delimiter,
            header=self.cfg.allocate.header,
            column_x=self.cfg.allocate.column_x,
            column_y=self.cfg.allocate.column_y,
            column_z=self.cfg.allocate.column_z,
            column_gene=self.cfg.allocate.column_gene,
            column_midcount=self.cfg.allocate.column_midcount,
            debug=self.cfg.allocate.debug,
        )

        log.info("Start allocation.")

        sdata = sp.tb.allocate(
            sdata=sdata,
            labels_layer=self.labels_layer_name,
            output_layer=self.cfg.allocate.table_layer_name,
            overwrite=self.cfg.allocate.overwrite,
        )

        log.info("Allocation finished.")

        sp.pl.plot_shapes(
            sdata,
            img_layer=self.cleaned_image_name,
            shapes_layer=self.shapes_layer_name,
            crd=self.cfg.segmentation.small_size_vis
            if self.cfg.segmentation.small_size_vis is not None
            else self.cfg.clean.small_size_vis,
            output=self.cfg.paths.polygons,
        )

        sp.pl.analyse_genes_left_out(
            sdata,
            labels_layer=self.labels_layer_name,
            table_layer=self.cfg.allocate.table_layer_name,
            output=self.cfg.paths.analyse_genes_left_out,
        )

        log.info("Preprocess AnnData.")

        # Perform preprocessing.
        sdata = sp.tb.preprocess_transcriptomics(
            sdata,
            labels_layer=self.labels_layer_name,
            table_layer=self.cfg.allocate.table_layer_name,
            output_layer=self.cfg.allocate.table_layer_name,
            min_counts=self.cfg.allocate.min_counts,
            min_cells=self.cfg.allocate.min_cells,
            size_norm=self.cfg.allocate.size_norm,
            n_comps=self.cfg.allocate.n_comps,
            overwrite=True,
        )

        log.info("Preprocessing AnnData finished.")

        sp.pl.preprocess_transcriptomics(
            sdata,
            table_layer=self.cfg.allocate.table_layer_name,
            output=self.cfg.paths.preprocess_adata,
        )

        sp.pl.plot_shapes(
            sdata,
            img_layer=self.cleaned_image_name,
            shapes_layer=self.shapes_layer_name,
            table_layer=self.cfg.allocate.table_layer_name,
            crd=self.cfg.segmentation.small_size_vis
            if self.cfg.segmentation.small_size_vis is not None
            else self.cfg.clean.small_size_vis,
            column=self.cfg.allocate.total_counts_column,
            cmap=self.cfg.allocate.total_counts_cmap,
            alpha=self.cfg.allocate.total_counts_alpha,
            output=self.cfg.paths.total_counts,
        )

        # Filter all cells based on size and distance
        sdata = sp.tb.filter_on_size(
            sdata,
            labels_layer=self.labels_layer_name,
            table_layer=self.cfg.allocate.table_layer_name,
            output_layer=self.cfg.allocate.table_layer_name,
            min_size=self.cfg.allocate.min_size,
            max_size=self.cfg.allocate.max_size,
            overwrite=True,
        )

        sp.pl.plot_shapes(
            sdata,
            img_layer=self.cleaned_image_name,
            shapes_layer=self.shapes_layer_name,
            table_layer=self.cfg.allocate.table_layer_name,
            crd=self.cfg.segmentation.small_size_vis
            if self.cfg.segmentation.small_size_vis is not None
            else self.cfg.clean.small_size_vis,
            column=_CELLSIZE_KEY,
            cmap=self.cfg.allocate.shape_size_cmap,
            alpha=self.cfg.allocate.shape_size_alpha,
            output=self.cfg.paths.shape_size,
        )

        log.info("Start clustering")

        sdata = sp.tb.leiden(
            sdata,
            labels_layer=self.labels_layer_name,
            table_layer=self.cfg.allocate.table_layer_name,
            output_layer=self.cfg.allocate.table_layer_name,
            calculate_umap=True,
            calculate_neighbors=True,
            n_pcs=self.cfg.allocate.pcs,
            n_neighbors=self.cfg.allocate.neighbors,
            resolution=self.cfg.allocate.cluster_resolution,
            rank_genes=True,
            key_added="leiden",
            overwrite=True,
        )

        log.info("Clustering finished")

        sp.pl.cluster(
            sdata,
            table_layer=self.cfg.allocate.table_layer_name,
            key_added="leiden",
            output=self.cfg.paths.cluster,
        )

        sp.pl.plot_shapes(
            sdata,
            img_layer=self.cleaned_image_name,
            shapes_layer=self.shapes_layer_name,
            table_layer=self.cfg.allocate.table_layer_name,
            column=self.cfg.allocate.leiden_column,
            cmap=self.cfg.allocate.leiden_cmap,
            alpha=self.cfg.allocate.leiden_alpha,
            crd=self.cfg.segmentation.small_size_vis
            if self.cfg.segmentation.small_size_vis is not None
            else self.cfg.clean.small_size_vis,
            output=self.cfg.paths.leiden,
        )

        if self.cfg.allocate.calculate_transcripts_density:
            # calculate transcript density
            sdata = sp.im.transcript_density(
                sdata,
                img_layer=self.cleaned_image_name,
                crd=self.cfg.segmentation.crop_param,
                scale_factors=self.cfg.dataset.scale_factors,
                output_layer=self.cfg.allocate.transcripts_density_img_layer_name,
                overwrite=self.cfg.allocate.overwrite,
            )

            sp.pl.transcript_density(
                sdata,
                img_layer=[
                    self.cleaned_image_name,
                    self.cfg.allocate.transcripts_density_img_layer_name,
                ],
                crd=self.cfg.segmentation.small_size_vis
                if self.cfg.segmentation.small_size_vis is not None
                else self.cfg.clean.small_size_vis,
                output=self.cfg.paths.transcript_density,
            )

        return sdata

    def annotate(self, sdata: SpatialData) -> SpatialData:
        """Annotation step, the fifth step of the pipeline, annotates the cells with celltypes based on the marker genes file."""
        # Get arguments from cfg else empty objects
        repl_columns = self.cfg.annotate.repl_columns if "repl_columns" in self.cfg.annotate else {}
        del_celltypes = self.cfg.annotate.del_celltypes if "del_celltypes" in self.cfg.annotate else []

        # Load marker genes, replace columns with different name, delete genes from list

        log.info("Start scoring genes")

        sdata, celltypes_scored, celltypes_all = sp.tb.score_genes(
            sdata=sdata,
            labels_layer=self.labels_layer_name,
            table_layer=self.cfg.allocate.table_layer_name,
            output_layer=self.cfg.allocate.table_layer_name,
            path_marker_genes=self.cfg.dataset.markers,
            delimiter=self.cfg.annotate.delimiter,
            row_norm=self.cfg.annotate.row_norm,
            repl_columns=repl_columns,
            del_celltypes=del_celltypes,
            overwrite=True,
        )

        self._celltypes = celltypes_all

        log.info("Scoring genes finished")

        sp.pl.score_genes(
            sdata=sdata,
            table_layer=self.cfg.allocate.table_layer_name,
            celltypes=celltypes_scored,  # celltypes_scored, is a list of all celltypes that are scored.
            shapes_layer=self.shapes_layer_name,
            img_layer=self.cleaned_image_name,
            output=self.cfg.paths.score_genes,
            crd=self.cfg.segmentation.small_size_vis
            if self.cfg.segmentation.small_size_vis is not None
            else self.cfg.clean.small_size_vis,
        )

        return sdata

    def visualize(self, sdata: SpatialData) -> SpatialData:
        """Visualisation step, the sixth and final step of the pipeline, checks the cluster cleanliness and performs nhood enrichement before saving the data as SpatialData object."""
        # Perform correction for transcripts (and corresponding celltypes) that occur in all cells and are overexpressed
        if "correct_marker_genes_dict" in self.cfg.visualize:
            sdata = sp.tb.correct_marker_genes(
                sdata,
                labels_layer=self.labels_layer_name,
                table_layer=self.cfg.allocate.table_layer_name,
                output_layer=self.cfg.allocate.table_layer_name,
                celltype_correction_dict=self.cfg.visualize.correct_marker_genes_dict,
                overwrite=True,
            )

        # Get arguments from cfg else None objects
        celltype_indexes = self.cfg.visualize.celltype_indexes if "celltype_indexes" in self.cfg.visualize else None
        colors = self.cfg.visualize.colors if "colors" in self.cfg.visualize else None

        # Check cluster cleanliness
        sdata, color_dict = sp.tb.cluster_cleanliness(
            sdata,
            labels_layer=self.labels_layer_name,
            table_layer=self.cfg.allocate.table_layer_name,
            output_layer=self.cfg.allocate.table_layer_name,
            celltypes=self._celltypes,
            celltype_indexes=celltype_indexes,
            colors=colors,
            overwrite=True,
        )

        sp.pl.cluster_cleanliness(
            sdata=sdata,
            table_layer=self.cfg.allocate.table_layer_name,
            img_layer=self.cleaned_image_name,
            shapes_layer=self.shapes_layer_name,
            crd=self.cfg.segmentation.small_size_vis
            if self.cfg.segmentation.small_size_vis is not None
            else self.cfg.clean.small_size_vis,
            color_dict=color_dict,
            output=self.cfg.paths.cluster_cleanliness,
        )

        # squidpy sometimes fails calculating/plotting nhood enrichement if a too small region is selected, therefore try add a try except.
        try:
            # calculate nhood enrichment
            sdata = sp.tb.nhood_enrichment(
                sdata,
                labels_layer=self.labels_layer_name,
                table_layer=self.cfg.allocate.table_layer_name,
                output_layer=self.cfg.allocate.table_layer_name,
                overwrite=True,
            )
            sp.pl.nhood_enrichment(
                sdata,
                table_layer=self.cfg.allocate.table_layer_name,
                output=self.cfg.paths.nhood,
            )
        except ValueError as e:
            log.warning(
                f"Could not calculate nhood enrichment for this region. Reason: {e}. Try with a different area if a subset was selected."
            )

        return sdata


def _check_config(cfg: DictConfig):
    """Checks if all paths and dataset paths are existing files, raise assertionError if not."""
    from pathlib import Path

    # Define the paths to check
    paths_to_check = [
        cfg.paths.data_dir,
        cfg.dataset.data_dir,
        cfg.dataset.coords,
        cfg.paths.output_dir,
    ]

    # If cfg.dataset.image is a list of paths, extend the paths_to_check with this list
    if isinstance(cfg.dataset.image, ListConfig):
        paths_to_check.extend(cfg.dataset.image)
    # Otherwise, just add the single path to paths_to_check
    else:
        paths_to_check.append(cfg.dataset.image)

    # Check if all mandatory paths exist
    for p in paths_to_check:
        # Check if the path contains a wildcard
        if "*" in p:
            matches = glob.glob(p)
            # Assert that at least one file matching the glob pattern exists
            assert matches, f"No file matches the path pattern {p}"
        else:
            assert Path(p).exists(), f"Path {p} does not exist."
