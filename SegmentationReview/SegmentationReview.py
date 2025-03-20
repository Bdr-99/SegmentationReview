import os

import vtk
import pathlib
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
# from QRCustomizations import CustomSegmentEditor # need to be installed seperate
import ctk
import qt
import time
import glob
import logging
import json
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

try:
    """Check if a package is installed in the Slicer environment space."""
    import pandas as pd
    import numpy as np
    import SimpleITK as sitk

except ImportError:
    # Install or import within the user Space (VMWare wont allow otherwise?)
    import importlib.util
    import site


    def is_installed(package):
        """Check if a package is installed in either the user space."""
        user_site = site.getusersitepackages()
        sys.path.append(user_site)  # add user space path
        found = importlib.util.find_spec(package) is not None
        return found


    def install(package, user=False):
        """Install package in Slicer environment or user space."""
        slicer.util.pip_install(f"--user {package}" if user else package)


    # List of required packages
    packages = ["pandas", "numpy", "SimpleITK"]

    for package in packages:
        if not is_installed(package):
            install(package, user=True)  # Try default install
            if not is_installed(package):
                raise "Could not import packages!"

    # Import packages after installation and/or adding user site space
    import pandas as pd
    import numpy as np
    import SimpleITK as sitk


#
# SegmentationReview
#

class SegmentationReview(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "PANCANCER SegmentationReview"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Anna Zapaishchykova (BWH), Dr. Benjamin H. Kann, AIM-Harvard"]
        self.parent.helpText = """
        This is an extension to help check the quality and find new lesions.<br><br>
        They are color coded:<br>
        - <span style='color:red;'>RED:</span> Human segmented<br>
        - <span style='color:blue;'>BLUE:</span> AI segmented<br>
        - <span style='color:green;'>GREEN:</span> Chosen segment of the two<br>
        - <span style='color:yellow;'>YELLOW:</span> Bounding box over new lesion<br><br>

        When you find a new lesion, please rename the lesion accordingly! Do this according to our segmentation guidelines.<br>
        <b>Link:</b><br>
        <a href='https://docs.google.com/document/d/1d-erzmi0oaTIRhf4RfPOoT7_8S9lGJCU/edit#heading=h.gjdgxs'>
        Segmentation Guidelines</a>
        """

        self.parent.acknowledgementText = """
This file was developed by AvL-NKI
"""


#
# SegmentationReviewWidget
#

class SegmentationReviewWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation

        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self.scan_node = None
        self.old_gt_seg_node = None
        self.ai_seg_node = None
        self.new_gt_seg_node = None

        # Default settings segmentations
        self.old_gt_seg_visible = False
        self.old_gt_seg_color = (0, 0, 1)  # We can change this
        self.ai_seg_visible = False
        self.ai_seg_color = (1, 1, 0)  # We can change this -> later on randomized
        self.new_gt_seg_visible = False
        self.new_gt_seg_color = (0, 1, 0)
        self.segmentation_index = 0  # index counter of which segmentation we are
        self.new_lesions = None  # for new lesions

        self.scan_dir = r'\\image-storage\RD_Radiogenomics\ct_lesion_detection\ct_scans'
        self.llama_dir = r'\\image-storage\RD_Radiogenomics\ct_lesion_detection\documentation\llama3_translated_improved'
        self.MEGA_documentation_path = r'\\image-storage\RD_Radiogenomics\ct_lesion_detection\documentation\datasets_tumtype.json'
        self.scan_files = []
        self.old_gt_seg_files = []
        self.ai_seg_files = []
        self.new_gt_seg_files = []

        self.directory = None
        self.current_index = 0

        self.current_segment_id = None
        self.likert_scores = []
        self.likert_scores_confidence = []
        self.n_files = 0
        self.current_df = None
        self.time_start = time.time()
        self.dummy_radio_buttons = []
        self.all_segments_visable = True  # Start with all the segments visable

        # ✅ Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # ✅ Ensure the logger only adds a handler once (avoid duplicates)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)  # Force logs to show in Slicer
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.propagate = False  # ✅ Prevents duplicate logs
        self._update_logging_level()  # ✅ Dynamically check test mode

        # ✅ Test if logging is working
        self.logger.debug("✅ Logger initialized successfully!")

    def _update_logging_level(self):
        """Dynamically updates logging level based on test mode."""
        if getattr(slicer, "test_mode", False):  # ✅ Check if `slicer.test_mode` exists
            self.logger.setLevel(logging.DEBUG)  # Set to DEBUG when running tests
        else:
            self.logger.setLevel(logging.INFO)  # Default to INFO in normal usage

    def log(self, level, message):
        """Wrapper function to ensure logging always uses the correct level."""
        self._update_logging_level()  # Ensure we always have the right log level
        if level == "debug":
            self.logger.debug(message)
        elif level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)


    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        import qSlicerSegmentationsModuleWidgetsPythonQt
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/SegmentationReview.ui'))

        # Layout within the collapsible button
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Input path"
        self.layout.addWidget(parametersCollapsibleButton)

        self.layout.addWidget(uiWidget)  # The annotation form
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        self.atlasDirectoryButton = ctk.ctkDirectoryButton()
        parametersFormLayout.addRow("Directory: ", self.atlasDirectoryButton)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = SlicerLikertDLratingLogic()
        self.layoutManager = slicer.app.layoutManager()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.PathLineEdit = ctk.ctkDirectoryButton()

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.atlasDirectoryButton.directoryChanged.connect(self.onAtlasDirectoryChanged)
        self.ui.save_and_next.connect('clicked(bool)', self.save_and_next_clicked)
        # self.ui.overwrite_mask.connect('clicked(bool)', self.overwrite_mask_clicked)
        self.ui.next_case.connect('clicked(bool)', self.next_case_clicked)
        self.ui.prev_case.connect('clicked(bool)', self.prev_case_clicked)
        self.ui.toggle_other_segmentations.clicked.connect(lambda: self.toggle_segments_visibility())
        self.ui.toggle_all.clicked.connect(lambda: self.toggle_all())
        self.ui.btnToggleSegmentationDisplay.clicked.connect(lambda: self.toggleSegmentationDisplay())
        self.ui.next_segmentation.connect('clicked(bool)', self.to_next_segment)
        self.ui.previous_segmentation.connect('clicked(bool)', self.to_previous_segment)
        self.ui.focus_segment.connect('clicked(bool)', self.jump_to_segmentation_slice)

        # self.ui.all_views.clicked.connect(lambda: self.change_orientation(3))
        # self.ui.transversal_view.clicked.connect(lambda: self.change_orientation(6))
        # self.ui.sagittal_view.clicked.connect(lambda: self.change_orientation(7))
        # self.ui.coronal_view.clicked.connect(lambda: self.change_orientation(8))

        self.ui.choose_seg_old.clicked.connect(lambda: self.choose_seg('old'))
        self.ui.choose_seg_ai.clicked.connect(lambda: self.choose_seg('ai'))
        self.ui.new_lesion.clicked.connect(lambda: self.new_lesion())

        # add a paint brush from segment editor window
        # Create a new segment editor widget and add it to the NiftyViewerWidget
        self._createSegmentEditorWidget_()

        defaultVolumeDisplayNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScalarVolumeDisplayNode")
        defaultVolumeDisplayNode.AutoWindowLevelOff()
        defaultVolumeDisplayNode.SetWindowLevelMinMax(-125, 225)
        slicer.mrmlScene.AddDefaultNode(defaultVolumeDisplayNode)
        # self.editorWidget.volumes.collapsed = True
        # Set parameter node first so that the automatic selections made when the scene is set are saved

        # Make sure parameter node is initialized (needed for module reload)
        # self.initializeParameterNode()

    def _createSegmentEditorWidget_(self):
        """Create and initialize a customize Slicer Editor which contains just some the tools that we need for the segmentation"""

        import qSlicerSegmentationsModuleWidgetsPythonQt

        # advancedCollapsibleButton
        self.segmentEditorWidget = qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget(
        )
        self.segmentEditorWidget.setMaximumNumberOfUndoStates(10)
        self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        self.segmentEditorWidget.unorderedEffectsVisible = False
        self.segmentEditorWidget.setEffectNameOrder([
            'Paint', 'Erase', 'Threshold',
        ])
        self.layout.addWidget(self.segmentEditorWidget)

    def get_base_names(self, folder_path, suffixes='.nrrd'):
        """
        Extracts base filenames by removing specified suffixes.

        Parameters:
        - folder_path (str): The path of the folder to scan.
        - suffixes (str or list): A single suffix as a string or a list of suffixes to remove.

        Returns:
        - dict: A dictionary where keys are base filenames and values are full file paths.
        """
        base_names = {}

        if isinstance(suffixes, str):
            suffixes = [suffixes]  # Convert to list if single suffix is provided

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if not os.path.isfile(file_path):
                continue  # Skip directories

            for suffix in suffixes:
                if filename.endswith(suffix):
                    base = filename.rsplit(suffix, 1)[0]
                    base_names[base] = file_path
                    break  # Stop at first matching suffix

        return base_names


    def add_additional_scans_to_map(self, folder_path, suffix='_0000.nrrd', scan_type="ct_scan", batch_size=500):
        """Efficiently adds additional scans to self.file_map while preserving existing scan types."""

        file_map_keys = set(self.file_map.keys())  # Convert keys to a set for O(1) lookup
        updates = {}  # Dictionary to store updates before applying batch update

        with os.scandir(folder_path) as entries:  # Faster than os.listdir()
            for entry in entries:
                if not entry.is_file():  # Skip directories
                    continue

                file_path = entry.path
                filename = entry.name

                # ✅ Handle `.txt` case separately
                if suffix == '.txt':
                    key_prefix = filename.split("-", 1)[0]  # Extract everything before the first '-'
                    matching_key = next((key for key in file_map_keys if key.startswith(key_prefix)), None)

                    if matching_key:
                        updates[matching_key] = {scan_type: file_path}  # Store in batch
                else:
                    # ✅ Handle regular suffix-based cases
                    if not filename.endswith(suffix):  # Skip non-matching files early
                        continue

                    key = filename.rsplit(suffix, 1)[0]  # Extract base name

                    if key in file_map_keys:  # Fast lookup
                        if key not in updates:
                            updates[key] = {}  # Ensure key exists in batch updates
                        updates[key][scan_type] = file_path  # Store in batch

                # ✅ Apply batch updates every `batch_size` items
                if len(updates) >= batch_size:
                    for key, new_data in updates.items():
                        self.file_map.setdefault(key, {}).update(new_data)  # Preserve old values
                    updates.clear()  # Reset batch

        # ✅ Apply remaining updates (even if < batch_size)
        if updates:
            for key, new_data in updates.items():
                self.file_map.setdefault(key, {}).update(new_data)  # Preserve old value

    def onAtlasDirectoryChanged(self, directory):
        """
        Initialized when directory is changed.
        Handles file extraction, filtering, and mapping for AI, GT, and CT files.
        """

        slicer.mrmlScene.Clear(0)  # 0 ensures the scene is cleared but not completely reset

        self.directory = directory
        self.reviewer_name = os.path.basename(self.directory)
        self.logger.info(f'-----------------------------------------------------')
        self.logger.info(f'Start looking for scans....')
        self.logger.info(f'Working in Folder: {self.reviewer_name}')
        self.time_start = time.time()

        # Initialize variables
        self.n_files = 0
        self.scan_files = []
        self.file_map = {}

        ### ✅ Step 1: Define Paths
        ct_path = self.scan_dir
        annotation_path = self.llama_dir
        ai_path = os.path.join(directory, 'ai')
        old_path = os.path.join(directory, 'old')
        new_path = os.path.join(directory, 'new/segments')
        roi_path = os.path.join(directory, 'new/roi')

        os.makedirs(new_path, exist_ok=True)
        os.makedirs(roi_path, exist_ok=True)

        ### ✅ Step 2: Get Base Names for AI & OLD Paths
        suffixes = ['.new.seg.nrrd', '.ai.seg.nrrd', '.gt.seg.nrrd', '.seg.nrrd', '_0000.nrrd', '.nrrd', '.txt']

        ai_base_names = self.get_base_names(ai_path, '.ai.seg.nrrd')  # {basename: full_path}
        old_base_names = self.get_base_names(old_path, '.gt.seg.nrrd')  # {basename: full_path}

        self.logger.debug(f"Found {len(ai_base_names)} AI segmentations")
        self.logger.debug(f"Found {len(old_base_names)} GT segmentations")

        self.logger.debug(f"Start searching for ai and old files")
        ### ✅ Step 3: Compare AI & OLD, create a `file_list` for common files
        self.file_map = {
            key: {"old": old_base_names[key], "ai": ai_base_names[key]}
            for key in sorted(set(ai_base_names.keys()) & set(old_base_names.keys()))
        }

        self.logger.debug(f"Found {len(self.file_map)} common segmentations between AI & OLD")
        self.logger.debug(f"Start initializing Pool for searching other files")

        ### ✅ Step 5: Get Additional Paths (New Segments, ROI, Annotations)
        with ThreadPoolExecutor() as executor:
            executor.submit(self.add_additional_scans_to_map, ct_path, '_0000.nrrd', 'ct_scan')
            executor.submit(self.add_additional_scans_to_map, new_path, '.new.seg.nrrd', 'new')
            executor.submit(self.add_additional_scans_to_map, roi_path, '.new_lesions.seg.nrrd', 'roi')
            executor.submit(self.add_additional_scans_to_map, annotation_path, '.txt', 'annotation')


        # Print all the different file types an how many we have of those
        scan_type_counts = {}
        for key in self.file_map:
            for scan_type in self.file_map[key]:  # Iterate over scan types present in this key
                if scan_type not in scan_type_counts:
                    scan_type_counts[scan_type] = 0
                scan_type_counts[scan_type] += 1  # Count occurrences

        self.logger.debug(f"Scan type counts: {scan_type_counts}")
        self.logger.info(f"Found {len(self.file_map)} scans inside the map")

        missing_segment_name = None

        for key in self.file_map:
            self.logger.debug(f"Already reviewed Scan: {key}")
            if self.file_map[key].get('new') is None:  # Avoid KeyError
                missing_segment_name = key  # Store key of first missing segmentation
                break  # Stop at the first missing segmentation

        # ✅ Step 8: Set Current Index for the First Missing Segmentation
        if missing_segment_name:
            for idx, (seg_name, data) in enumerate(self.file_map.items()):
                if seg_name == missing_segment_name:
                    self.current_index = idx
                    break
            self.logger.info(f"First missing segmentation '{missing_segment_name}' found at index {self.current_index}")
        else:
            self.logger.info("No missing segmentation found. Continuing from the last checked segmentation.")

        with open(self.MEGA_documentation_path, "r") as file:
            self.MEGA_documentation = json.load(file)

        ### ✅ Step 9: Finalize & Update UI
        self.n_files = len(self.file_map)
        self.ui.status_checked.setText(f"Checked Scans: {self.current_index} / {self.n_files - 1}")
        self.resetUIElements()

        self.logger.debug(f'Finished initializing Pool for searching other files')

        end_time = time.time()
        self.logger.debug(f'Time taken: {round(end_time - self.time_start, 2)} seconds')

        ### ✅ Step 10: Load First File with Mask
        self.load_all()
        self.time_start = time.time()

    def save_and_next_clicked(self):
        """
        Saves the current segmentation and ROI, then proceeds to the next case.
        If some GT segments are missing from `self.segments_stats`, prompts a warning before saving.
        Also saves `self.segments_stats` and file paths to a CSV file using pandas.
        """

        # Check if all Old GT segment IDs exist in self.segments_stats
        missing_segments = []
        old_gt_segmentation = self.old_gt_seg_node.GetSegmentation()

        for i in range(old_gt_segmentation.GetNumberOfSegments()):
            segment_id = old_gt_segmentation.GetNthSegmentID(i)
            if segment_id not in self.segments_stats:
                missing_segments.append(segment_id)

        # Determine message based on missing segments
        msg_text = (
            "<br><span style='color:red;'>[WARNING]</span> <br>Some GT segments are not inside the new segmentation. Are you sure you want to save it?"
            if missing_segments else "Are you sure you want to go to the next one?"
        )
        # Show confirmation popup
        msg = qt.QMessageBox()
        msg.setIcon(qt.QMessageBox.Warning)
        msg.setText(msg_text)
        msg.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)
        response = msg.exec_()

        if response == qt.QMessageBox.No:
            if missing_segments:
                # Jump to the first missing segment
                self.segmentation_index = next(
                    (index for index, segment_id in enumerate(self.segment_ids) if segment_id in missing_segments),
                    self.segmentation_index  # Default to current index if no match is found
                )
                self.current_segment_id = self.segment_ids[self.segmentation_index]
                self._ensure_current_segment_visible()  # Ensure visibility of the missing segment
                self.jump_to_segmentation_slice()
                self.update_segment_availability_status()
                self.logger.warning(f"Jumping to first missing segment: {self.current_segment_id}")
            return  # Stop execution if the user selects 'No'

        # Get current scan key
        scan_key = self.scan_key  # Unique identifier for the scan

        # Retrieve existing file paths from file_map
        file_entry = self.file_map.get(scan_key, {})

        # Construct paths
        new_seg_path = os.path.join(self.directory, "new", "segments", f"{self.new_gt_seg_node.GetName()}.seg.nrrd")

        roi_path = None  # Default to None
        if self.ROI_segmentation_node is not None:
            roi_path = os.path.join(self.directory, "new", "roi", f"{self.ROI_segmentation_node.GetName()}.seg.nrrd")

        csv_path = os.path.join(self.directory, f"segment_stats_{self.reviewer_name}.csv")  # CSV for segmentation stats

        # Save segmentation & ROI (only if ROI exists)
        slicer.util.saveNode(self.new_gt_seg_node, new_seg_path)
        if roi_path:
            slicer.util.saveNode(self.ROI_segmentation_node, roi_path)

        # Save self.segments_stats to CSV using pandas
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(columns=["scan_name", "segment_id", "choice", "label_type", "segment_comment", "new_seg",
                                       "new_seg_path", "roi_path"])

        # Check if scan_key already exists in CSV
        existing_rows = df[df["scan_name"] == scan_key]

        if not existing_rows.empty:
            # Ask if user wants to overwrite existing scan data
            if self._confirm_action("Already existing segmentation data, do you want to overwrite it?"):
                df = df[df["scan_name"] != scan_key]  # Remove old data

        # Create new DataFrame for the current segmentation with file paths
        new_data = pd.DataFrame([
            {
                "scan_name": scan_key,
                "segment_id": segment_id,
                "choice": stats.get("choice", ""),
                "label_type": stats.get("label_type", ""),
                "segment_comment": stats.get("segment_comment", ""),
                "new_seg": stats.get("new_seg", ""),
                "new_seg_path": new_seg_path,
                "roi_path": roi_path  # If no ROI segmentation node, this will be None
            }
            for segment_id, stats in self.segments_stats.items()
        ])

        # Append new data and save CSV
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_csv(csv_path, index=False)

        # Clear segment_stats from memory to prevent OOM
        self.segments_stats.clear()
        self.logger.info(f"Segment statistics saved to {csv_path}")

        # ✅ Update self.file_map with new paths
        self.file_map[scan_key] = {
            "ct_scan": file_entry.get("ct_scan"),
            "old": file_entry.get("old"),
            "ai": file_entry.get("ai"),
            "new": new_seg_path,
            "roi": roi_path  # Set to None if no ROI segmentation node
        }

        # Delete existing model storage nodes so that they will be recreated with default settings
        existingModelStorageNodes = slicer.util.getNodesByClass("vtkMRMLModelStorageNode")
        for modelStorageNode in existingModelStorageNodes:
            slicer.mrmlScene.RemoveNode(modelStorageNode)

        # Move to next case
        self.current_index += 1
        self.load_all()
        self.time_start = time.time()
        self.ui.status_checked.setText(f"Checked Scans: {self.current_index} / {self.n_files - 1}")
        self.resetUIElements()
        self.ui.segment_comment.setPlainText("")

    def next_case_clicked(self):
        # A check if we checked all segmentations???

        msg = qt.QMessageBox()
        msg.setIcon(qt.QMessageBox.Warning)
        msg.setText(f"You sure you want to go to the next one?\n All progress will be lost")
        msg.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)
        response = msg.exec_()
        if response == qt.QMessageBox.No:
            return  # Stop execution if the user selects 'No'
        # Delete existing model storage nodes so that they will be recreated with default settings
        existingModelStorageNodes = slicer.util.getNodesByClass("vtkMRMLModelStorageNode")
        for modelStorageNode in existingModelStorageNodes:
            slicer.mrmlScene.RemoveNode(modelStorageNode)

        self.current_index += 1
        self.load_all()
        self.time_start = time.time()
        self.ui.status_checked.setText("Checked Scans: " + str(self.current_index) + " / " + str(self.n_files - 1))
        self.resetUIElements()
        self.ui.segment_comment.setPlainText("")

    def prev_case_clicked(self):
        if self.current_index != 0:
            msg = qt.QMessageBox()
            msg.setIcon(qt.QMessageBox.Warning)
            msg.setText(f"You sure you want to go to the previous one?\n All progress will be lost")
            msg.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)
            response = msg.exec_()
            if response == qt.QMessageBox.No:
                return  # Stop execution if the user selects 'No'

            existingModelStorageNodes = slicer.util.getNodesByClass("vtkMRMLModelStorageNode")
            for modelStorageNode in existingModelStorageNodes:
                slicer.mrmlScene.RemoveNode(modelStorageNode)

            self.current_index -= 1
            self.load_all()
            self.time_start = time.time()
            self.ui.status_checked.setText("Checked Scans: " + str(self.current_index) + " / " + str(self.n_files - 1))
            self.resetUIElements()
            self.ui.segment_comment.setPlainText("")

    def get_likert_score_from_ui(self, radio_buttons):
        for idx, radio_button in enumerate(radio_buttons, 1):
            if radio_button.isChecked():
                return idx
        return 0

    def load_all(self):
        """
        Loads the scan, old ground truth segmentation, and AI segmentation for the current index.
        Creates an empty segmentation node for the new ground truth segmentation.
        Removes all nodes from the scene before loading.
        Extracts the number of unique segmentation places and updates the status QLabel.
        Jumps to the slice view of the first segmentation index.
        """
        # ✅ Step 1: Clear Scene & Reset Variables
        slicer.mrmlScene.Clear(0)  # 0 ensures the scene is cleared but not completely reset
        slicer.util.resetSliceViews()
        self.logger.info('-----------------------------------------------------')
        self.logger.info("Scene cleared. Loading new data...")

        self.all_segments_visable = True
        self.segments_stats = {}
        self.segment_names = []
        self.segment_ids = []
        self.segmentation_index = 0
        self.old_seg_available = False
        self.ai_seg_available = False
        self.ROI_index = 0
        self.ROI_segmentation_node = None
        self.scan_key = None

        # ✅ Step 2: Retrieve File Paths for Current Index
        self.scan_key = list(self.file_map.keys())[self.current_index]  # Get the corresponding key
        scan_key = self.scan_key
        file_data = self.file_map[scan_key]  # Retrieve all associated paths
        scan_id = scan_key.split("-")[0]

        ct_scan_path = file_data['ct_scan']
        old_gt_seg_path = file_data['old']
        ai_seg_path = file_data['ai']
        new_gt_seg_path = file_data.get('new', None)  # Might be missing
        roi_path = file_data.get('roi', None)  # Might be missing
        annotation_path = file_data.get('annotation', None)  # Might be missing

        self.logger.debug(f"Loading Scan: {scan_key}")
        self.logger.debug(f"CT Path: {ct_scan_path}")
        self.logger.debug(f"Old GT Path: {old_gt_seg_path}")
        self.logger.debug(f"AI Segmentation Path: {ai_seg_path}")
        self.logger.debug(f"New GT Path: {new_gt_seg_path}")
        self.logger.debug(f"ROI Path: {roi_path}")
        self.logger.debug(f"Annotation Path: {annotation_path}")

        # ✅ Step 3: Load CT Scan
        if ct_scan_path:
            self.scan_node = slicer.util.loadVolume(ct_scan_path)
            self.scan_node.SetName(scan_key)
            slicer.app.applicationLogic().PropagateVolumeSelection(0)
        else:
            self.logger.error(f"❌ Missing CT scan for {scan_key}")

        # ✅ Step 4: Load Old GT Segmentation
        if old_gt_seg_path:
            self.old_gt_seg_node = slicer.util.loadSegmentation(old_gt_seg_path)
            if self.old_gt_seg_node:
                old_gt_display_node = self.old_gt_seg_node.GetDisplayNode()
                old_gt_display_node.SetVisibility2DFill(False)
                old_gt_display_node.SetVisibility2DOutline(True)
                old_gt_display_node.SetColor(self.old_gt_seg_color)
                old_gt_display_node.SetVisibility(True)

                segmentation = self.old_gt_seg_node.GetSegmentation()
                for i in range(segmentation.GetNumberOfSegments()):
                    self.segment_ids.append(segmentation.GetNthSegmentID(i))
                    self.segment_names.append(segmentation.GetSegment(segmentation.GetNthSegmentID(i)).GetName())
                    segment_id = segmentation.GetNthSegmentID(i)
                    segment = segmentation.GetSegment(segment_id)
                    segment.SetColor(1, 0, 0)  # RED
        else:
            self.logger.warning(f"⚠️ Missing Old GT Segmentation for {scan_key}")

        # ✅ Step 5: Load AI Segmentation
        if ai_seg_path:
            self.ai_seg_node = slicer.util.loadSegmentation(ai_seg_path)
            if self.ai_seg_node:
                ai_display_node = self.ai_seg_node.GetDisplayNode()
                ai_display_node.SetVisibility2DFill(False)
                ai_display_node.SetVisibility2DOutline(True)
                # ai_display_node.SetColor(0, 0, 1)
                ai_display_node.SetVisibility(True)

                segmentation = self.ai_seg_node.GetSegmentation()
                for i in range(segmentation.GetNumberOfSegments()):
                    segment_id = segmentation.GetNthSegmentID(i)
                    segment = segmentation.GetSegment(segment_id)
                    segment.SetColor(0, 0, 1)  # RGB
                    if segmentation.GetNthSegmentID(i) not in self.segment_ids:
                        self.segment_ids.append(segmentation.GetNthSegmentID(i))
                        self.segment_names.append('NOT DEFINED YET')
        else:
            self.logger.warning(f"⚠️ Missing AI Segmentation for {scan_key}")

        self.logger.debug(self.segment_ids)
        self.logger.debug(self.segment_names)

        # ✅ Step 6: Load or Create New GT Segmentation
        if new_gt_seg_path:
            self.new_gt_seg_node = slicer.util.loadSegmentation(new_gt_seg_path)
            if self.new_gt_seg_node:
                self.update_comment('Existing Segmentation file loaded in', 'orange')
                self.logger.info(f'Found existing new segmentation file for {scan_id}, loading it in.')
        else:
            self.new_gt_seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{scan_key}.new")
            self.new_gt_seg_node.CreateDefaultDisplayNodes()
            self.update_comment(f'Segmentation file created for {scan_id}.')

        new_gt_display_node = self.new_gt_seg_node.GetDisplayNode()
        new_gt_display_node.SetVisibility2DFill(False)
        new_gt_display_node.SetVisibility2DOutline(True)
        new_gt_display_node.SetColor(self.new_gt_seg_color)
        new_gt_display_node.SetVisibility(True)

        # ✅ Step 7: Load ROI (If Available)
        if roi_path:
            self.ROI_segmentation_node = slicer.util.loadSegmentation(roi_path)
            if self.ROI_segmentation_node:
                ROI_segmentation_node = self.ROI_segmentation_node.GetDisplayNode()
                ROI_segmentation_node.SetVisibility2DFill(False)
                ROI_segmentation_node.SetVisibility2DOutline(True)
                ROI_segmentation_node.SetVisibility(True)
        else:
            self.logger.debug(f"Missing ROI for {scan_key}")

        # ✅ Step 8: Load Annotation form (If Available)
        if annotation_path:
            with open(annotation_path, "r", encoding="utf-8") as file:
                annotation_text = file.read()  # Store the text content
                self.ui.annotation_form.setPlainText(annotation_text)

        else:
            annotation_text = next(
                (self.MEGA_documentation[dataset] for dataset in self.MEGA_documentation if dataset in scan_key),
                "No Radiology Report Found")
            self.ui.annotation_form.setHtml(annotation_text)

        slicer.app.processEvents()  # Allow UI to catch up

        # ✅ Step 9: Display the Number of Segments
        num_segments = len(self.segment_ids)
        self.current_segment_id = self.segment_ids[0] if self.segment_ids else None

        # ✅ Step 10: Finalize UI
        self.jump_to_segmentation_slice()
        self.toggle_segments_visibility(True)
        self.update_segment_availability_status()
        self.set_segmentation_and_mask_for_segmentation_editor()

        self.logger.info("✅ Done Loading in!.")
        self.logger.info(f"Total unique segmentation places: {num_segments}")
        end_time = time.time()
        self.logger.debug(f'Time taken: {round(end_time - self.time_start, 2)} seconds')
        self.logger.info('-----------------------------------------------------\n')

    def choose_seg(self, choice):
        """
        Copies a selected segment from either the AI segmentation or Old GT segmentation
        into the new GT segmentation, ensuring correct naming and tracking statistics.

        If the segment already exists in the new GT segmentation, it is overwritten.
        If a segment is removed, its corresponding stats are also deleted.

        Tracks:
        - 'new_seg': True if AI is the only available source and chosen, otherwise False.
        - 'name': The segment's assigned name.
        - 'label_type': Always set to 'strong'.
        - 'segment_comment': Stores user-provided comment (can be empty).

        Args:
            choice (str): 'ai' or 'old', indicating the source segmentation.
        """
        self.logger.debug(f'Segmentation chosen: {choice}')

        new_segmentation = self.new_gt_seg_node.GetSegmentation()
        segment_id = self.current_segment_id
        segment_exists = new_segmentation.GetSegment(segment_id) is not None
        source_segmentation = None

        # Get the user comment (can be empty)
        segment_comment = self.ui.segment_comment.toPlainText().strip() if self.ui.segment_comment else ""

        # Initialize stats entry for this segment (overwrite if re-choosing)
        self.segments_stats[segment_id] = {
            "choice": choice,
            "label_type": "strong",  # Always set to 'strong'
            "segment_comment": segment_comment  # Save the comment, even if empty
        }
        if self.old_seg_available:
            # Set segmentation name if GT is available
            segment_name = self.old_gt_seg_node.GetSegmentation().GetSegment(segment_id).GetName()

        # Determine source segmentation and update stats
        # If both GT and AI are available
        if self.ai_seg_available and self.old_seg_available:
            source_segmentation = self.old_gt_seg_node.GetSegmentation() if choice == 'old' else self.ai_seg_node.GetSegmentation()
            self.segments_stats[segment_id]["new_seg"] = False  # GT segment exists, so not new
            # Get the GT segment name
        # If only AI is available:
        elif self.ai_seg_available:
            if choice == 'ai':
                source_segmentation = self.ai_seg_node.GetSegmentation()
                self.segments_stats[segment_id]["new_seg"] = True  # True if no GT segment exists
                # Get new segment name if AI is new
                if not segment_exists:
                    segment_name = self.get_segment_name()
                    if segment_name is None:
                        self.logger.info('No name chosen. Choose again')
                        del self.segments_stats[segment_id]  # Cleanup
                        return
                    self.segments_stats[segment_id]["name"] = segment_name
            # AI segment already exists and you dont want to keep it
            else:
                self.segments_stats.pop(segment_id, None)  # Remove stats entry
                if not segment_exists:
                    self.logger.debug(f'No lesion, going to the next without saving')
                else:
                    self.logger.debug(f'Lesion already inside the new segment. Removing it...')
                    new_segmentation.RemoveSegment(segment_id)
                self.to_next_segment()
                return

        # Only GT is available
        elif self.old_seg_available:
            # Keep GT segment
            if choice == 'old':
                source_segmentation = self.old_gt_seg_node.GetSegmentation()
                self.segments_stats[segment_id]["new_seg"] = False  # Old GT is not a new segment
            else:
                # Removing GT? Not really a good option??
                self.logger.debug('REMOVING GT?')
                self._remove_segment_with_confirmation(new_segmentation, segment_id,
                                                       "Remove already existing GT segmentation?")
                return
        else:
            self.logger.warning("No segmentation available to copy or remove.")
            del self.segments_stats[segment_id]  # Cleanup
            return

        # If a segment exists, confirm overwrite
        if segment_exists:
            if not self._confirm_action(f"Segment already exists. Overwrite?"):
                return
            new_segmentation.RemoveSegment(segment_id)

        # Copy segment from the chosen source
        if source_segmentation and source_segmentation.GetSegment(segment_id):
            new_segmentation.CopySegmentFromSegmentation(source_segmentation, segment_id, False)
            new_segmentation.GetSegment(segment_id).SetName(segment_name)
            new_segmentation.GetSegment(segment_id).SetColor(0, 1, 0)
            self.logger.debug(f"✅ Segment {segment_name} copied from {choice}.")

            # Ensure stats are updated correctly
            self.segments_stats[segment_id]["name"] = segment_name
            self.segments_stats[segment_id]["label_type"] = "strong"  # Always 'strong'
            self.segments_stats[segment_id]["segment_comment"] = segment_comment  # Save the comment

        self.to_next_segment()

    def _remove_segment_with_confirmation(self, segmentation, segment_id, message):
        """
        Removes a segment from the given segmentation after user confirmation and cleans up stats.

        Args:
            segmentation (vtkSegmentation): The segmentation to modify.
            segment_id (str): The ID of the segment to remove.
            message (str): The confirmation message.
        """
        # If segmentation in new exists
        if segmentation.GetSegment(segment_id):
            if self._confirm_action(message):
                segmentation.RemoveSegment(segment_id)
                self.segments_stats.pop(segment_id, None)  # Remove stats entry
                self.to_next_segment()
        # To handle the skipping of segments
        else:
            if self._confirm_action(message):
                self.to_next_segment()
                self.segments_stats.pop(segment_id, None)  # Remove stats entry

    def _confirm_action(self, message):
        """
        Displays a confirmation dialog with Yes/No options.

        Args:
            message (str): The message to display.

        Returns:
            bool: True if the user confirms, False otherwise.
        """
        msg = qt.QMessageBox()
        msg.setIcon(qt.QMessageBox.Warning)
        msg.setText(message)
        msg.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)
        return msg.exec_() == qt.QMessageBox.Yes

    def update_comment(self, message, color="white"):
        """
        Updates the UI message box with a given message and applies a color.

        Args:
            message (str): The message to display.
            color (str): The text color ("red", "orange", "black", etc.).
        """
        self.ui.message.setTextFormat(qt.Qt.RichText)  # Enable rich text formatting
        self.ui.message.setText(f"<p style='color:{color}; font-weight:bold;'>{message}</p>")

    def change_orientation(self, orientation=3):
        """
        Change the orientation layout viewer
        :param orientation:
            - 3: All 3
            - 6: Transversal
            - 7: Sagital
            - 8: Coronal
            - 29: Transversal + Sagital
        :return:
        """
        self.logger.debug(f'Orientation clicked!: {orientation}')
        self.layoutManager.setLayout(orientation)

    def put_it_asside(self):
        """
        For when i dont really know for sure if the lesions are correct and want to have a radiologist take a look at it
        :return:
        """
        ## DO: make function to save current index case to other folder for later reviewing with kalina

    def resetUIElements(self):
        # Check all dummy radio buttons to effectively uncheck the other buttons in the group
        for dummy_rb in self.dummy_radio_buttons:
            dummy_rb.setChecked(True)

        # Reset the comment section
        self.ui.segment_comment.setPlainText("")
        self.logger.debug("All UI elements reset.")

    def jump_to_segmentation_slice(self):
        """
        Centers the slice views and cameras on the selected segment.
        Handles cases where the segment exists only in AI or only in Old GT.
        """

        position = None  # Default value

        # Check if the segment exists in Old GT
        if self.old_gt_seg_node and self.old_gt_seg_node.GetSegmentation().GetSegment(self.current_segment_id):
            position = self.old_gt_seg_node.GetSegmentCenterRAS(self.current_segment_id)

        # If not found in Old GT, check AI segmentation
        elif self.ai_seg_node and self.ai_seg_node.GetSegmentation().GetSegment(self.current_segment_id):
            position = self.ai_seg_node.GetSegmentCenterRAS(self.current_segment_id)

        # Center slice views and cameras on this position
        for sliceNode in slicer.util.getNodesByClass('vtkMRMLSliceNode'):
            sliceNode.JumpSliceByCentering(*position)

        for camera in slicer.util.getNodesByClass('vtkMRMLCameraNode'):
            camera.SetFocalPoint(position)

        self.toggle_all(True)  # Only showing the one segmentation
        slicer.app.processEvents()

    def toggleSegmentationDisplay(self, forceVisibility=None):
        """
        Toggle between fill and outline display modes for all segmentation nodes.

        Optionally, force a specific mode:
        - `forceVisibility=True` → Enable Fill mode.
        - `forceVisibility=False` → Enable Outline mode.
        - `forceVisibility=None` (default) → Toggle between Fill and Outline.

        Args:
            forceVisibility (bool, optional): If True, sets Fill mode;
                                              if False, sets Outline mode;
                                              if None, toggles normally.
        """

        segmentationNodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        if not segmentationNodes:
            self.logger.warning("No segmentation nodes found.")
            return

        # Check current state from first segmentation node
        displayNode = segmentationNodes[0].GetDisplayNode()
        currentFillVisibility = displayNode.GetVisibility2DFill()

        if forceVisibility is None:
            newFillVisibility = not currentFillVisibility  # Toggle Fill
        else:
            newFillVisibility = forceVisibility  # Force Fill or Outline mode

        # Apply visibility settings to all segmentations
        for segNode in segmentationNodes:
            displayNode = segNode.GetDisplayNode()
            if displayNode:
                displayNode.SetVisibility2DFill(newFillVisibility)
                displayNode.SetVisibility2DOutline(not newFillVisibility)

        # Update the button text accordingly
        self.ui.btnToggleSegmentationDisplay.setText("Show Fill" if not newFillVisibility else "Show Outline")
        self.logger.debug(f"Segmentation Display Updated: Fill={newFillVisibility}, Outline={not newFillVisibility}")
        slicer.app.processEvents()

    def toggle_segments_visibility(self, toggle=None):
        """
        Toggle between showing all segments or only the selected segment
        in all segmentation nodes.

        Parameters:
        - toggle (bool, optional): If True, forces all segments to be visible.
                                   If False, forces only the selected segment to be visible.
                                   If None (default), toggles the current state.
        """

        segmentationNodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        # Show all segmentations
        self.toggle_all(True)
        if not segmentationNodes:
            self.logger.warning("No segmentation nodes found.")
            return

        # Get current toggle state
        if toggle is None:
            toggle = not self.all_segments_visable  # Flip the current state

        for segNode in segmentationNodes:
            segmentation = segNode.GetSegmentation()
            displayNode = segNode.GetDisplayNode()

            if not displayNode:
                continue  # Skip if display node is missing

            # Show all segments
            if toggle:
                for i in range(segmentation.GetNumberOfSegments()):
                    segment_id = segmentation.GetNthSegmentID(i)
                    displayNode.SetSegmentVisibility(segment_id, True)
            else:
                # Show only the selected segment
                for i in range(segmentation.GetNumberOfSegments()):
                    segment_id = segmentation.GetNthSegmentID(i)
                    displayNode.SetSegmentVisibility(segment_id, segment_id == self.current_segment_id)

        # Update button text
        self.ui.toggle_other_segmentations.setText("Only Selected Segment" if toggle else "Show All Segments")
        self.all_segments_visable = toggle  # Update state
        self.logger.debug(f"{'Showing all segments' if toggle else 'Showing only selected segment'}")
        slicer.app.processEvents()

    def toggle_all(self, toggle=None):
        """
        Toggle visibility for all segmentation nodes in the scene.

        Parameters:
        - toggle (bool, optional): If True, forces all segmentations to be visible.
                                   If False, hides all segmentations.
                                   If None (default), toggles the current state.
        """

        # Get all segmentation nodes in the scene
        segmentationNodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")

        if not segmentationNodes:
            self.logger.warning("No segmentation nodes found.")
            return

        # Check current visibility state (assuming all segmentations have the same state)
        currentVisibility = segmentationNodes[0].GetDisplayNode().GetVisibility()

        # If toggle is None, flip the current state
        if toggle is None:
            toggle = not currentVisibility

        # Apply visibility change to all segmentation nodes
        for segNode in segmentationNodes:
            displayNode = segNode.GetDisplayNode()
            if displayNode:
                displayNode.SetVisibility(toggle)

        # Print status
        self.logger.debug(f"{'Showing' if toggle else 'Hiding'} all segmentation nodes.")
        slicer.app.processEvents()

    def to_next_segment(self):
        """
        Moves to the next segment while ensuring its visibility.
        """
        try:
            segment_stats = self.segments_stats.get(self.current_segment_id, None)
            if segment_stats is not None:
                self.logger.debug(f"Saved segment stats: {segment_stats}")
            else:
                self.logger.warning(f"No segment stats found for segment ID: {self.current_segment_id}")
        except Exception as e:
            self.logger.error(f"Error retrieving segment stats: {e}")

        if self.segmentation_index + 1 >= len(self.segment_ids):
            self.logger.warning("No more segments available!")
            self.ui.next_segmentation.setText("No more segments!")
            self.update_comment('No More Segmentations!', 'orange')
            self.ui.choose_seg_old.setText(f'❌')
            self.ui.choose_seg_ai.setText(f'❌')
            self.ui.choose_seg_old.setStyleSheet("background-color: black; color: white;")
            self.ui.choose_seg_ai.setStyleSheet("background-color: black; color: white;")
            self.ui.segmentation_text.setText(f'"❌ No more segments available! ❌"')
            return

        self.segmentation_index += 1
        self.current_segment_id = self.segment_ids[self.segmentation_index]
        self.ui.next_segmentation.setText("Next (NO SAVE)")

        # Ensure the new segment is visible
        self._ensure_current_segment_visible()

        # Maintain existing functionality
        self.jump_to_segmentation_slice()
        self.update_segment_availability_status()

    def to_previous_segment(self):
        """
        Moves to the previous segment while ensuring its visibility.
        """
        if not self.segmentation_index == 0:
            try:
                segment_stats = self.segments_stats.get(self.current_segment_id, None)
                if segment_stats is not None:
                    self.logger.debug(f"Saved segment stats: {segment_stats}")
                else:
                    self.logger.warning(f"No segment stats found for segment ID: {self.current_segment_id}")
            except Exception as e:
                self.logger.error(f"Error retrieving segment stats: {e}")

            # Move to previous segment
            self.segmentation_index -= 1

            self.current_segment_id = self.segment_ids[self.segmentation_index]
            self.ui.next_segmentation.setText("Next (NO SAVE)")

            # Ensure the new segment is visible
            self._ensure_current_segment_visible()

            # Maintain existing functionality
            self.jump_to_segmentation_slice()
            self.update_segment_availability_status()

    def _ensure_current_segment_visible(self):
        """
        Ensures the current segment is visible in all segmentation nodes.
        """

        segmentationNodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        if not segmentationNodes:
            self.logger.warning("No segmentation nodes found.")
            return

        for segNode in segmentationNodes:
            segmentation = segNode.GetSegmentation()
            display_node = segNode.GetDisplayNode()

            if not display_node:
                self.logger.warning(f"Segmentation node '{segNode.GetName()}' has no display node.")
                continue  # Skip nodes without display settings

            # Ensure visibility for the current segment in this segmentation
            if segmentation.GetSegment(self.current_segment_id):
                display_node.SetSegmentVisibility(self.current_segment_id, True)
                # self.logger.debug(
                #    f"✅ Ensured visibility for segment '{self.current_segment_id}' in '{segNode.GetName()}'.")
            # else:
            # self.logger.debug(f"⚠️ Segment '{self.current_segment_id}' not found in '{segNode.GetName()}'.")
        slicer.app.processEvents()

    def update_segment_availability_status(self):
        """
        Checks whether the segment exists in AI and/or Old GT segmentations and returns the formatted UI status string.
        """
        ai_exists = self.ai_seg_node and self.ai_seg_node.GetSegmentation().GetSegment(
            self.segment_ids[self.segmentation_index]) is not None
        old_gt_exists = self.old_gt_seg_node and self.old_gt_seg_node.GetSegmentation().GetSegment(
            self.segment_ids[self.segmentation_index]) is not None

        if ai_exists and old_gt_exists:
            segment_source_text = "Segment Available: 2 ✅"
            self.ui.choose_seg_old.setText(f'Segmentation 1')
            self.ui.choose_seg_ai.setText(f'Segmentation 2')
            self.ui.choose_seg_old.setStyleSheet("background-color: red; color: white;")
            self.ui.choose_seg_ai.setStyleSheet("background-color: blue; color: white;")
            self.ui.segmentation_text.setText(f'Choose a segmentation:')
            self.old_seg_available = True
            self.ai_seg_available = True
        elif ai_exists:
            segment_source_text = "Segment Available: ✅ AI ❌ Old GT"
            self.ui.choose_seg_old.setStyleSheet("background-color: black; color: white;")
            self.ui.choose_seg_ai.setStyleSheet("background-color: blue; color: white;")
            self.ui.choose_seg_old.setText(f'NO LESION ❌')
            self.ui.choose_seg_ai.setText(f'NEW LESION ✅')
            self.ui.segmentation_text.setText(f'New lesion?')
            self.old_seg_available = False
            self.ai_seg_available = True
        elif old_gt_exists:
            segment_source_text = "Segment Available: ❌ AI ✅ Old GT"
            self.ui.choose_seg_old.setText(f'Keep Segmentation ✅')
            self.ui.choose_seg_ai.setText(f'Remove Segmentation ❌')
            self.ui.choose_seg_old.setStyleSheet("background-color: red; color: white;")
            self.ui.choose_seg_ai.setStyleSheet("background-color: black; color: white;")
            self.ui.segmentation_text.setText(f'Only 1 Segmentation available')
            self.old_seg_available = True
            self.ai_seg_available = False
        else:
            segment_source_text = "Segment Available: ❌ AI ❌ Old GT"
            self.ui.choose_seg_old.setText(f'❌')
            self.ui.choose_seg_ai.setText(f'❌')
            self.ui.segmentation_text.setText(f'"❌ No more segments available! ❌"')
            self.old_seg_available = False
            self.ai_seg_available = False

        text = f"Segmentation: {self.segmentation_index + 1} / {len(self.segment_names)} Name: {self.segment_names[self.segmentation_index]}"
        # Get full UI status text
        self.ui.status_segments.setText(text)
        slicer.app.processEvents()

        try:
            segment_comment = self.segments_stats.get(self.current_segment_id, {}).get("segment_comment", "")
            self.ui.segment_comment.setPlainText(segment_comment)
        except Exception as e:
            self.logger.debug(f"Failed to load segment comment")
            self.ui.segment_comment.setPlainText("")
        return

    def new_lesion(self):
        """
        Creates a new ROI, allows user to place it, and detects when the size changes.
        """

        # Ensure no existing ROI is currently active (to prevent duplication)
        existing_rois = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsROINode")
        existing_rois.UnRegister(None)  # Prevent memory leaks

        if existing_rois.GetNumberOfItems() > 0:
            self.create_segment_from_roi()
            return

        # Generate ROI name dynamically
        roi_name = f"NEW_LESION_{self.ROI_index}"

        # Create a new ROI node
        self.roi_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", roi_name)
        self.roi_node.SetDisplayVisibility(True)
        self.logger.info(f"Created ROI: {roi_name}")

        # Enable placement mode
        slicer.modules.markups.logic().SetActiveListID(self.roi_node)
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetPlaceModePersistence(0)  # Allow placement mode
        interactionNode.SetCurrentInteractionMode(slicer.vtkMRMLInteractionNode.Place)
        self.ui.new_lesion.setText(f'CONFIRM BOUNDING BOX')
        self.ui.choose_seg_old.setText(f'❌')
        self.ui.choose_seg_ai.setText(f'❌')
        self.ui.choose_seg_old.setStyleSheet("background-color: black; color: white;")
        self.ui.choose_seg_ai.setStyleSheet("background-color: black; color: white;")
        self.ui.choose_seg_old.blockSignals(True)
        self.ui.choose_seg_ai.blockSignals(True)

        self.ui.segmentation_text.setText(f'Confirm Bounding Box')

    def create_segment_from_roi(self):
        """
        Converts the ROI into a segment of a segmentation node and removes the ROI node after conversion.
        Prompts the user to name the new segment.

        Additionally, it stores segment statistics:
        - 'name': The segment's assigned name.
        - 'label_type': Always set to 'weak'.
        - 'segment_comment': Stores user-provided comment (can be empty).
        - 'new_seg': Always True (as this is a new segment).
        - 'choice': Set to 'self' (indicating it was manually created).
        """
        if not self.roi_node:
            self.logger.warning("❌ No ROI node found. Create an ROI first.")
            return

        # Check if the ROI is placed (valid bounds)
        bounds = [0] * 6
        self.roi_node.GetRASBounds(bounds)

        # If bounds are too small or unmodified, the ROI hasn't been placed
        if all(abs(b) < 0.01 for b in bounds):
            self.logger.warning("❌ ROI has not been placed. Please place the ROI before converting.")
            return

        # Find reference volume by name
        reference_volume = slicer.mrmlScene.GetFirstNodeByName(self.scan_key)
        if not reference_volume:
            self.logger.warning(f"No reference volume found with name: {self.scan_key}")
            return

        # Ask the user for a segment name
        segment_name = self.get_segment_name()
        self.logger.debug(segment_name)
        if not segment_name:
            self.logger.warning("Segment creation canceled by user.")
            slicer.mrmlScene.RemoveNode(self.roi_node)
            self.roi_node = None
            self.update_segment_availability_status()
            self.ui.new_lesion.setText(f'Found new lesion')
            return

        # Convert ROI to model representation
        roi_model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "ROI_Model")
        cube = vtk.vtkCubeSource()
        cube.SetBounds(bounds)
        cube.Update()
        roi_model_node.SetAndObservePolyData(cube.GetOutput())
        roi_model_node.CreateDefaultDisplayNodes()

        if self.ROI_segmentation_node is None:  # Creation of the ROI segmentation node
            self.ROI_segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode",
                                                                            f"{self.scan_key}.new_lesions")
            self.ROI_segmentation_node.CreateDefaultDisplayNodes()  # Ensure it has proper display settings
            ROI_segmentation_node = self.ROI_segmentation_node.GetDisplayNode()
            ROI_segmentation_node.SetVisibility(True)
            ROI_segmentation_node.SetVisibility2DFill(False)
            ROI_segmentation_node.SetVisibility2DOutline(True)
            self.ROI_segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.scan_node)

        # Import the model as a new segment in the segmentation node
        slicer.modules.segmentations.logic().ImportModelToSegmentationNode(roi_model_node, self.ROI_segmentation_node)

        # Set the segment name
        self.ROI_index += 1  # Increment index for next ROI

        segmentation = self.ROI_segmentation_node.GetSegmentation()
        segment_id = segmentation.GetNthSegmentID(segmentation.GetNumberOfSegments() - 1)  # Get the last added segment
        self.logger.debug(f'ROI segment ID: {segment_id}')
        segmentation.GetSegment(segment_id).SetName(segment_name)
        self.logger.debug(f'NEW set NAME: {segmentation.GetNthSegmentID(segmentation.GetNumberOfSegments() - 1)}')

        # Check if the segment is empty by exporting it to a labelmap
        labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(self.ROI_segmentation_node, [segment_id],
                                                                          labelmap)

        # Validate if the segment contains data
        image_data = labelmap.GetImageData()
        if not image_data or image_data.GetScalarRange() == (0.0, 0.0):  # No non-zero values = Empty
            self.logger.warning(f"❌ Segment '{segment_name}' is empty. Removing it.")
            segmentation.RemoveSegment(segment_id)  # Remove the empty segment
        else:
            self.logger.debug(f"✅ Segment '{segment_name}' contains data and was kept.")

            # Retrieve segment comment (even if empty)
            segment_comment = self.ui.segment_comment.toPlainText().strip() if self.ui.segment_comment else ""

            # Save segment stats
            self.segments_stats[segment_id] = {
                "name": segment_name,
                "label_type": "weak",
                "segment_comment": segment_comment,
                "new_seg": True,
                "choice": "self"
            }
            self.logger.debug(f"📊 Stats updated for {segment_id}: {self.segments_stats[segment_id]}")

        # Clean up the temporary model
        slicer.mrmlScene.RemoveNode(roi_model_node)

        # Remove the ROI node
        slicer.mrmlScene.RemoveNode(self.roi_node)
        slicer.mrmlScene.RemoveNode(labelmap)
        self.roi_node = None

        self.logger.debug(f"✅ ROI converted into segment '{segment_name}' and ROI node removed.")
        self.ui.new_lesion.setText(f'Found new lesion')
        self.ui.choose_seg_old.blockSignals(False)
        self.ui.choose_seg_ai.blockSignals(False)
        self.update_segment_availability_status()

    def get_segment_name(self):
        """
        Opens a dialog to ask the user for a segment name.
        Returns the name as a string or None if canceled.
        """
        self.logger.debug('Starting up dialog for segment name.')
        result = qt.QInputDialog.getText(None, "Segment Name", "Enter name for the new segment:", qt.QLineEdit.Normal)
        self.logger.debug(f'OUTPUT NAME: {result}')
        # Ensure result is unpacked properly
        if len(result) > 0:
            self.logger.debug(f'User entered segment name: {result}')
            return result
        self.logger.debug('Segment naming canceled or empty input')
        return None  # Return None if canceled

    def set_segmentation_and_mask_for_segmentation_editor(self):
        slicer.app.processEvents()
        if not hasattr(self, 'segmentEditorWidget') or self.segmentEditorWidget is None:
            import qSlicerSegmentationsModuleWidgetsPythonQt
            self.segmentEditorWidget = qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget()
            self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)

        self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
        slicer.mrmlScene.AddNode(segmentEditorNode)
        self.segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
        self.segmentEditorWidget.setSegmentationNode(self.new_gt_seg_node)
        self.segmentEditorWidget.setSourceVolumeNode(self.scan_node)
        self.segmentEditorWidget.setActiveEffectByName("Threshold")
        effect = self.segmentEditorWidget.activeEffect()
        if effect:
            effect.setParameter("MinimumThreshold", -50)
            effect.setParameter("MaximumThreshold", 200)
        self.segmentEditorWidget.setActiveEffectByName("")

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        # if inputParameterNode:
        #    self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.EndModify(wasModified)


#
# SlicerLikertDLratingLogic
#

class SlicerLikertDLratingLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)


#
# SlicerLikertDLratingTest
#

class SlicerLikertDLratingTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """

        self.setUp()
        self.test_SlicerLikertDLrating1()

    def test_SlicerLikertDLrating1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        self.delayDisplay('Test passed')


class SegmentationReviewTest(ScriptedLoadableModuleTest):
    """
    This test initializes SegmentationReviewWidget and automatically sets the atlas directory,
    without opening a new UI window.
    """

    def setUp(self):
        """ Reset the scene before running the test. """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """ Run the test. """
        self.setUp()
        self.test_reload_and_set_directory()

    def test_reload_and_set_directory(self):
        """
        Simulates module reload and sets the directory automatically.
        """

        self.delayDisplay("Starting SegmentationReview initialization test")

        # **Step 1**: Reload the module as if the user pressed "Reload"
        slicer.util.reloadScriptedModule("SegmentationReview")

        # **Step 2**: Get the module widget
        widget = slicer.modules.SegmentationReviewWidget

        if not widget:
            self.fail("SegmentationReviewWidget could not be initialized.")

        slicer.test_mode = True

        # **Step 3**: Set the directory and trigger function
        test_directory = r"\\image-storage\RD_Radiogenomics\ct_lesion_detection\reviewer_TEST"
        widget.onAtlasDirectoryChanged(test_directory)

        # **Step 4**: Ensure the directory was correctly set
        self.assertEqual(widget.directory, test_directory, "Directory was not set correctly!")

        self.delayDisplay("SegmentationReview initialization test passed")