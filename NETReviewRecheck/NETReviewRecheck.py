import logging
import os

import vtk
import pathlib
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import ctk
import qt
import time
import glob

try:
    import pandas as pd
    import numpy as np
    import SimpleITK as sitk

except:
    slicer.util.pip_install('pandas')
    slicer.util.pip_install('numpy')
    slicer.util.pip_install('SimpleITK')

    import pandas as pd
    import numpy as np
    import SimpleITK as sitk


import pandas as pd
import os


def load_mappings(csv_file, excel_file):
    # Read the CSV file
    df_csv = pd.read_csv(csv_file)
    # Create a dictionary mapping from New File and Destination Folder to Original File
    # Using a tuple (New File, Destination Folder) as the key
    mapping_new_dest_to_original = {(row['New File'], row['Destination Folder']): row['Original File'] for _, row in df_csv.iterrows()}

    # Read the Excel file
    df_excel = pd.read_csv(excel_file)
    # Create a dictionary mapping from Anonymized Name to Patient ID
    mapping_anonymized_to_id = {row['Anonymized Name']: row['Patient ID'] for _, row in df_excel.iterrows()}

    return mapping_new_dest_to_original, mapping_anonymized_to_id


def find_patient_info(new_file, destination_folder):
    # Links
    root = r'\\image-storage\RD CRC-data\_archive\IRBd23231-AMONET\code'
    csv_file = os.path.join(root, 'selected_scans.csv')
    excel_file = os.path.join(root, '20230926 JADS Export Report Patients.csv')

    mapping_new_dest_to_original, mapping_anonymized_to_id = load_mappings(csv_file, excel_file)

    # Find the original file using a tuple of new file and destination folder
    original_file = mapping_new_dest_to_original.get((new_file, destination_folder))
    if not original_file:
        return "Original file not found for given New File and Destination Folder"

    # Extract anonymized patient identifier
    anonymized_name = original_file.split('\\')[1]

    # Extract exam date
    exam_date = original_file.split('\\')[2].split(' ')[0]

    # Find the patient ID
    patient_id = mapping_anonymized_to_id.get(anonymized_name)
    if not patient_id:
        return "Patient ID not found for given Anonymized Name", "-"

    return patient_id, exam_date


def update_annotations_csv(updated_row_data, directory, filename="NET_annotations.csv"):
    # Load the existing annotations into a DataFrame
    file_path = os.path.join(directory, filename)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        # If the file doesn't exist, create a new DataFrame with the same structure as `updated_row_data`
        df = pd.DataFrame(columns=updated_row_data.keys())
    
    # Check if the file we are updating already exists in the DataFrame
    file_index = df[df['file'] == updated_row_data['file']].index
    if file_index.empty:
        # If the file does not exist, append the new row
        df = df.append(updated_row_data, ignore_index=True)
    else:
        # If the file exists, update the row with the new data
        for key, value in updated_row_data.items():
            df.at[file_index[0], key] = value

    # Write the DataFrame back to the CSV, overwriting the original file
    df.to_csv(file_path, index=False)

#
# NETReviewRecheck
#

class NETReviewRecheck(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "NETReview (Recheck)"
        self.parent.categories = ["Examples"]
        self.parent.dependencies = []
        self.parent.contributors = ["Anna Zapaishchykova (BWH), Dr. Benjamin H. Kann, AIM-Harvard"]
        self.parent.helpText = """
Slicer3D extension for rating using Likert-type score Deep-learning generated segmentations, with segment editor funtionality. 
Created to speed up the validation process done by a clinician - the dataset loads in one batch with no need to load masks and volumes separately.
It is important that each nii file has a corresponding mask file with the same name and the suffix _mask.nii
"""

        self.parent.acknowledgementText = """
This file was developed by Anna Zapaishchykova, BWH. 
"""


#
# NETReviewRecheckWidget
#

class NETReviewRecheckWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        self.volume_node = None
        self.segmentation_node = None
        self.segmentation_visible = False
        self.segmentation_color = [1, 0, 0]
        self.nifti_files = []
        self.segmentation_files = []
        self.directory = None
        self.current_index = 0
        self.likert_scores = []
        self.likert_scores_confidence = []
        self.n_files = 0
        self.current_df = None
        self.time_start = time.time()
        self.groupCheckableGroups = []

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        import qSlicerSegmentationsModuleWidgetsPythonQt
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/NETReviewRecheck.ui'))

        # Layout within the collapsible button
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Input path"
        self.layout.addWidget(parametersCollapsibleButton)

        self.layout.addWidget(uiWidget)
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

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.PathLineEdit = ctk.ctkDirectoryButton()

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.atlasDirectoryButton.directoryChanged.connect(     self.onAtlasDirectoryChanged)
        self.ui.btnNothingToSegment.connect('clicked(bool)',    self.onNothingToSegmentClicked)
        self.ui.btnNoLiverToSegment.connect('clicked(bool)',    self.onNoLiverToSegmentClicked)
        self.ui.btnNoContrast.connect('clicked(bool)',          self.onNoContrastClicked)
        self.ui.btnSaveAndNext.connect('clicked(bool)',         self.onSaveAndNextClicked)
        self.ui.btnNextCase.connect('clicked(bool)',            self.onNextCaseClicked)
        self.ui.btnPrevCase.connect('clicked(bool)',            self.onPreviousCaseClicked)
        self.ui.btnToggleSegmentationDisplay.clicked.connect(   self.toggleSegmentationDisplay)

        self.groupCheckableGroups = [
            self.ui.groupBoxLiverImaged,
            self.ui.groupBoxImageQuality,
            self.ui.groupBoxContrast,
            self.ui.groupBoxPhaseTiming,
            self.ui.groupBoxAISegmentation,
            self.ui.groupBoxManualSegmentation,
            self.ui.groupBoxDifficultySegmentation
        ]

        # add a paint brush from segment editor window
        # Create a new segment editor widget and add it to the NiftyViewerWidget
        self._createSegmentEditorWidget_()

        defaultVolumeDisplayNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScalarVolumeDisplayNode")
        defaultVolumeDisplayNode.AutoWindowLevelOff()
        defaultVolumeDisplayNode.SetWindowLevel(200, 100)
        slicer.mrmlScene.AddDefaultNode(defaultVolumeDisplayNode)
        slicer.app.applicationLogic().GetInteractionNode().SetCurrentInteractionMode(slicer.vtkMRMLInteractionNode.ViewTransform)

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

    def overwrite_mask_clicked(self):
        # Assume segmentation_file_path stores the original .seg.nii.gz file path
        segmentation_file_path = self.segmentation_files[self.current_index]
        
        # For some reason it doesnt save nifti files directly
        # so we first save in nrrd, then load the nrrd with simpleitk, then save it in nifti
        segmentation_file_path = segmentation_file_path.replace('.seg.nii.gz', '.seg.nrrd')
        slicer.util.saveNode(self.segmentation_node, segmentation_file_path)

        img = sitk.ReadImage(segmentation_file_path)
        nrrd_to_delete = segmentation_file_path
        segmentation_file_path = segmentation_file_path.replace('.seg.nrrd', '.seg.nii.gz')
        sitk.WriteImage(img, segmentation_file_path)
        os.remove(nrrd_to_delete)

    def onAtlasDirectoryChanged(self, directory):
        if not os.path.isfile(directory + "/NET_annotations.csv"):
            print("NET_annotations.csv not found in the directory!")
            print("Did you lead the correct directory?")
            print("Did you mean to load the other pug-in for first time case revision?")
            return
        
        self.first_reader_annotations = pd.read_csv(directory + "/NET_annotations.csv")

        if self.volume_node:
            slicer.mrmlScene.RemoveNode(self.volume_node)
        if self.segmentation_node:
            slicer.mrmlScene.RemoveNode(self.segmentation_node)

        self.directory = directory

        # Initialize these variables at the beginning
        self.n_files = 0
        self.nifti_files = []
        self.segmentation_files = []

        # Load the existing annotations if the file exists
        annotated_files = set()
        if os.path.exists(directory + "/NET_annotations_2.csv"):
            self.current_df = pd.read_csv(directory + "/NET_annotations_2.csv")
            annotated_files = set(self.current_df['file'].values)

        else:
            columns = [
                'file', 'is_liver_imaged', 'contrast', 'phase_timing', 'no_lesions',
                'ai_segmentation_quality', 'manual_segmentation_confidence', 
                'adjustment_segmentation_difficulty', 'high_signal_to_noise', 
                'metal_artifacts', 'patient_motion', 'other', 'comment', 'time-stamp'
            ]
            self.current_df = pd.DataFrame(columns=columns)
        
        self.current_index = 0

        # Collect images and masks, skipping already annotated ones
        for seg_file in glob.glob(directory + '/*.seg.nii.gz'):
            file = seg_file.replace(".seg", "")
            filename = file.split(os.sep)[-1]

            # Skip the file if it's already annotated
            if filename in annotated_files:
                continue

            self.n_files += 1
            self.nifti_files.append(file)
            self.segmentation_files.append(seg_file)


        # count the number of files in the directory
        self.resetUIElements()

        # load first file with mask
        self.load_first_reviewer_reply()
        self.load_nifti_file()
        self.time_start = time.time()

    def onNoContrastClicked(self):
        # Logic to save the file path and set the contrast phase to "no contrast"
        # Similar to the existing methods

        new_row = {
            'file': self.nifti_files[self.current_index].split(os.sep)[-1],
            'contrast': 'NO_CONTRAST',
            'comment': self.ui.comment.toPlainText(),
            'time-stamp': time.time()
        }

        update_annotations_csv(new_row, self.directory)
        self.processNextCase()  

    def onNothingToSegmentClicked(self):

        # Collect responses from the UI
        liver_imaged_score = self.get_likert_score_from_ui([
            self.ui.radioLiverYes,
            self.ui.radioLiverNo
        ])

        contrast_score = self.get_likert_score_from_ui([
            self.ui.radioButtonContrastArterial,
            self.ui.radioButtonContrastPortal,
            self.ui.radioButtonContrastLatePhase
        ])

        phase_timing_score = self.get_likert_score_from_ui([
            self.ui.radioButtonPhaseTooEarly,
            self.ui.radioButtonPhaseJustRight,
            self.ui.radioButtonPhaseTooLate
        ])

        new_row = {
            'file': self.nifti_files[self.current_index].split(os.sep)[-1],
            'is_liver_imaged': liver_imaged_score,
            'contrast': contrast_score,
            'phase_timing': phase_timing_score,
            'high_signal_to_noise': self.ui.checkNoiseRatio.isChecked(),
            'metal_artifacts': self.ui.checkMetalArtifacts.isChecked(),
            'patient_motion': self.ui.checkPatientMotion.isChecked(),
            'no_lesions': 'TRUE',
            'other': self.ui.lineEditOtherReasons.text,
            'comment': self.ui.comment.toPlainText(),
            'time-stamp': time.time()
        }

        update_annotations_csv(new_row, self.directory)
        self.processNextCase()  

    def onNoLiverToSegmentClicked(self):

        new_row = {
            'file': self.nifti_files[self.current_index].split(os.sep)[-1],
            'is_liver_imaged': 'NO_LIVER',
            'comment': self.ui.comment.toPlainText(),
            'time-stamp': time.time()
        }

        update_annotations_csv(new_row, self.directory)
        self.processNextCase()  

    def onSaveAndNextClicked(self):
        # Generic category (assuming it's kept the same for reference)

        if not self.all_responses_provided():
            print("Please provide all required responses before proceeding.")
            return

        # Collect responses from the UI
        liver_imaged_score = self.get_likert_score_from_ui([
            self.ui.radioLiverYes,                  # 1
            self.ui.radioLiverNo                    # 2
        ])

        contrast_score = self.get_likert_score_from_ui([
            self.ui.radioButtonContrastArterial,    # 1
            self.ui.radioButtonContrastPortal,      # 2
            self.ui.radioButtonContrastLatePhase    # 3
        ])

        phase_timing_score = self.get_likert_score_from_ui([
            self.ui.radioButtonPhaseTooEarly,       # 1
            self.ui.radioButtonPhaseJustRight,      # 2
            self.ui.radioButtonPhaseTooLate         # 3
        ])

        ai_segmentation_quality_score = self.get_likert_score_from_ui([
            self.ui.radioButtonAISegmentation1,
            self.ui.radioButtonAISegmentation2,
            self.ui.radioButtonAISegmentation3,
            self.ui.radioButtonAISegmentation4,
            self.ui.radioButtonAISegmentation5
        ])

        manual_segmentation_confidence_score = self.get_likert_score_from_ui([
            self.ui.radioButtonManualSegmentation1,
            self.ui.radioButtonManualSegmentation2,
            self.ui.radioButtonManualSegmentation3,
            self.ui.radioButtonManualSegmentation4,
            self.ui.radioButtonManualSegmentation5
        ])

        adjustment_segmentation_difficulty_score = self.get_likert_score_from_ui([
            self.ui.radioButtonDifficultySegmentation1,
            self.ui.radioButtonDifficultySegmentation2,
            self.ui.radioButtonDifficultySegmentation3,
            self.ui.radioButtonDifficultySegmentation4,
            self.ui.radioButtonDifficultySegmentation5
        ])

        new_row = {
            'file': self.nifti_files[self.current_index].split(os.sep)[-1],
            'is_liver_imaged': liver_imaged_score,
            'contrast': contrast_score,
            'phase_timing': phase_timing_score,
            'ai_segmentation_quality': ai_segmentation_quality_score,
            'manual_segmentation_confidence': manual_segmentation_confidence_score,
            'adjustment_segmentation_difficulty': adjustment_segmentation_difficulty_score,
            'high_signal_to_noise': self.ui.checkNoiseRatio.isChecked(),
            'metal_artifacts': self.ui.checkMetalArtifacts.isChecked(),
            'patient_motion': self.ui.checkPatientMotion.isChecked(),
            'other': self.ui.lineEditOtherReasons.text,
            'comment': self.ui.comment.toPlainText(),
            'time-stamp': time.time()
        }

        update_annotations_csv(new_row, self.directory)
        self.overwrite_mask_clicked()
        self.processNextCase()  

    def processNextCase(self):
        if self.current_index < self.n_files - 1:
            self.current_index += 1
            self.load_nifti_file()
            self.resetUIElements()
            self.load_first_reviewer_reply()
            self.time_start = time.time()
        else:
            self.ui.status_checked.setText("All files are checked!!")
            print("All files checked")

    def onNextCaseClicked(self):
        if self.current_index < self.n_files - 1:
            self.current_index += 1
            self.load_nifti_file()
            self.resetUIElements()
            self.load_first_reviewer_reply()
            self.time_start = time.time()

    def onPreviousCaseClicked(self):
        if self.current_index != 0:
            self.current_index -= 1
            self.load_nifti_file()
            self.resetUIElements()
            self.load_first_reviewer_reply()
            self.time_start = time.time()

    def toggleSegmentationDisplay(self):
        if not self.segmentation_node:
            print("Segmentation node is not loaded yet!")
            return

        displayNode = self.segmentation_node.GetDisplayNode()

        if not displayNode:
            print("Segmentation doesn't have a display node!")
            return

        # Check current display mode
        currentMode = displayNode.GetVisibility2DFill()

        if currentMode:  # If it's currently in fill mode
            displayNode.SetVisibility2DFill(False)  # Set fill off
            displayNode.SetVisibility2DOutline(True)  # Set outline on
            self.ui.btnToggleSegmentationDisplay.setText("Show Fill")
        else:  # If it's currently in outline mode
            displayNode.SetVisibility2DFill(True)  # Set fill on
            displayNode.SetVisibility2DOutline(False)  # Set outline off
            self.ui.btnToggleSegmentationDisplay.setText("Show Outline")

    def get_likert_score_from_ui(self, radio_buttons):
        for idx, radio_button in enumerate(radio_buttons, 1):
            if radio_button.isChecked():
                return idx
        return 0

    def load_nifti_file(self):
        # Reset the slice views to clear any remaining segmentations
        slicer.util.resetSliceViews()

        file_path = self.nifti_files[self.current_index]
        if self.volume_node:
            slicer.mrmlScene.RemoveNode(self.volume_node)
        if self.segmentation_node:
            slicer.mrmlScene.RemoveNode(self.segmentation_node)

        self.volume_node = slicer.util.loadVolume(file_path)
        slicer.app.applicationLogic().PropagateVolumeSelection(0)

        segmentation_file_path = self.segmentation_files[self.current_index]
        self.segmentation_node = slicer.util.loadSegmentation(segmentation_file_path)

        # Setting the visualization of the segmentation to outline only
        segmentationDisplayNode = self.segmentation_node.GetDisplayNode()
        segmentationDisplayNode.SetVisibility2DFill(True)  # Do not show filled region in 2D
        segmentationDisplayNode.SetVisibility2DOutline(True)  # Show outline in 2D
        segmentationDisplayNode.SetColor(self.segmentation_color)
        segmentationDisplayNode.SetVisibility(True)

        self.set_segmentation_and_mask_for_segmentation_editor()

        # Normalize the path to convert any forward slashes and double backslashes 
        # to the standard os-specific path separator
        normalized_path = os.path.normpath(file_path)
        # Split the path into its components
        path_components = normalized_path.split(os.sep)
        # Get filename and batch 
        filename, batch = path_components[-1], path_components[-2]
        try:
            pt, exam = find_patient_info(filename, batch)
            
            # Format date string
            date = pd.to_datetime(exam, format='%Y%m%d')
            formatted_date = date.strftime('%dth %b %Y')

            day = date.day
            if 4 <= day <= 20 or 24 <= day <= 30:
                suffix = "th"
            else:
                suffix = ["st", "nd", "rd"][day % 10 - 1]

            formatted_date = formatted_date.replace('th', suffix)
        except:
            pt              = "error loading"
            formatted_date  = "error loading"

        self.ui.patient_id.setText(pt)
        self.ui.exam_date.setText(formatted_date)

        display_str = "Checked: " + str(self.current_index) + " / " + str(self.n_files - 1)
        self.ui.status_checked.setText(display_str)

        # check if correction is already present 
        correction = self.segmentation_files[self.current_index].replace(".seg.nii.gz", ".seg_2.nii.gz")
        if os.path.exists(correction):
            self.ui.status_checked.setText(display_str + " [THIS CASE WAS ALREADY CORRECTED! Showing original readout.]")

        print(file_path, segmentation_file_path)

    def resetUIElements(self):
        # Check all dummy radio buttons to effectively uncheck the other buttons in the group
        self.ui.radioLiverHidden.setChecked(True)
        self.ui.radioButtonContrastHidden.setChecked(True)
        self.ui.radioButtonPhaseHidden.setChecked(True)
        self.ui.radioButtonAISegmentationHidden.setChecked(True)
        self.ui.radioButtonManualSegmentationHidden.setChecked(True)
        self.ui.radioButtonDifficultySegmentationHidden.setChecked(True)
        self.ui.checkNoiseRatio.setChecked(False)
        self.ui.checkMetalArtifacts.setChecked(False)
        self.ui.checkPatientMotion.setChecked(False)
        self.ui.checkOtherReasons.setChecked(False)
        self.ui.lineEditOtherReasons.setText("")
        self.ui.comment.setPlainText("")
        self.ui.btnNoContrast.setStyleSheet("QPushButton { color: black; }")
        self.ui.btnNoLiverToSegment.setStyleSheet("QPushButton { color: black; }")
        self.ui.btnNothingToSegment.setStyleSheet("QPushButton { color: black; }")
        print("All UI elements reset.")

    def setUIElements(
            self, 
            liver_imaged_score=None, 
            contrast_score=None, 
            phase_timing_score=None, 
            ai_segmentation_quality_score=None, 
            manual_segmentation_confidence_score=None, 
            adjustment_segmentation_difficulty_score=None, 
            high_signal_to_noise=False, 
            metal_artifacts=False, 
            patient_motion=False, 
            other='', 
            comment=''
        ):
        # Helper function to set the radio button group
        def set_radio_buttons(button_group, score):
            print('setting ', button_group, ' to value ', score)
            if score is None or score == 0 or pd.isna(score):
                getattr(self.ui, button_group + 'Hidden').setChecked(True)
            else:
                score = int(score) if isinstance(score, float) else score
                getattr(self.ui, button_group + str(score)).setChecked(True)

        # Set radio buttons
        set_radio_buttons('radioLiver', liver_imaged_score)
        set_radio_buttons('radioButtonContrast', contrast_score)
        set_radio_buttons('radioButtonPhase', phase_timing_score)
        set_radio_buttons('radioButtonAISegmentation', ai_segmentation_quality_score)
        set_radio_buttons('radioButtonManualSegmentation', manual_segmentation_confidence_score)
        set_radio_buttons('radioButtonDifficultySegmentation', adjustment_segmentation_difficulty_score)

        # Set checkboxes
        self.ui.checkNoiseRatio.setChecked(high_signal_to_noise)
        self.ui.checkMetalArtifacts.setChecked(metal_artifacts)
        self.ui.checkPatientMotion.setChecked(patient_motion)

        # Set text inputs
        self.ui.lineEditOtherReasons.setText(other)
        self.ui.comment.setPlainText(comment)

    def load_first_reviewer_reply(self):
        file_name = os.path.basename(self.nifti_files[self.current_index])

        # Find the record matching the file_name
        record = self.first_reader_annotations[self.first_reader_annotations['file'] == file_name]

        if not record.empty:
            print(record.squeeze())

            def to_int_or_nan(value):
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return float('nan')
                
            def string_to_bool(input_value):
                if isinstance(input_value, np.bool_):
                    input_value = bool(input_value)
                if isinstance(input_value, bool):
                    # If the input is already a boolean, return it as is
                    return input_value
                elif isinstance(input_value, str):
                    # If the input is a string, check for "true" or "false" (case-insensitive)
                    lower_input = input_value.lower()
                    if lower_input == 'true':
                        return True
                    elif lower_input == 'false':
                        return False

                # If input is neither a boolean nor a recognizable string, return None or raise an exception
                return None
            
            if record['contrast'].iloc[0] == 'NO_CONTRAST':
                self.ui.btnNoContrast.setStyleSheet("QPushButton { color: red; }")
            if record['is_liver_imaged'].iloc[0] == 'NO_LIVER':
                self.ui.btnNoLiverToSegment.setStyleSheet("QPushButton { color: red; }")
            if string_to_bool(record['no_lesions'].iloc[0]):
                self.ui.btnNothingToSegment.setStyleSheet("QPushButton { color: red; }")

            # Extract values from the record
            liver_imaged_score                              = to_int_or_nan(record['is_liver_imaged'].iloc[0])
            contrast_score                                  = to_int_or_nan(record['contrast'].iloc[0])
            phase_timing_score                              = to_int_or_nan(record['phase_timing'].iloc[0])
            ai_segmentation_quality_score                   = to_int_or_nan(record['ai_segmentation_quality'].iloc[0])
            manual_segmentation_confidence_score            = to_int_or_nan(record['manual_segmentation_confidence'].iloc[0])
            adjustment_segmentation_difficulty_score        = to_int_or_nan(record['adjustment_segmentation_difficulty'].iloc[0])
            high_signal_to_noise                            = string_to_bool(record['high_signal_to_noise'].iloc[0])
            metal_artifacts                                 = string_to_bool(record['metal_artifacts'].iloc[0])
            patient_motion                                  = string_to_bool(record['patient_motion'].iloc[0])
            other                                           = record['other'].iloc[0] if not pd.isna(record['other'].iloc[0]) else ''
            comment                                         = record['comment'].iloc[0] if not pd.isna(record['comment'].iloc[0]) else ''

            # fix the entries that are stored as number but are not a number
            liver_imaged_score      = 'Yes' if liver_imaged_score == 1 else 'No' if liver_imaged_score == 2 else None
            contrast_score          = 'Arterial' if contrast_score == 1 else 'Portal' if contrast_score == 2 else 'LatePhase' if contrast_score == 3 else None
            phase_timing_score      = 'TooEarly' if phase_timing_score == 1 else 'JustRight' if phase_timing_score == 2 else 'TooLate' if phase_timing_score == 3 else None

            # Call setUIElements with the extracted values
            self.setUIElements(
                liver_imaged_score, 
                contrast_score, 
                phase_timing_score, 
                ai_segmentation_quality_score, 
                manual_segmentation_confidence_score, 
                adjustment_segmentation_difficulty_score, 
                high_signal_to_noise, 
                metal_artifacts, 
                patient_motion, 
                other, 
                comment
            )
        else:
            print(f"No record found for file name: {file_name}")

    def set_segmentation_and_mask_for_segmentation_editor(self):
        slicer.app.processEvents()
        self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
        slicer.mrmlScene.AddNode(segmentEditorNode)
        self.segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
        self.segmentEditorWidget.setSegmentationNode(self.segmentation_node)
        self.segmentEditorWidget.setSourceVolumeNode(self.volume_node)

    def all_responses_provided(self):
        # List of all dummy radio buttons

        # Check if any dummy radio button is checked
        for dummy_rb in self.groupCheckableGroups:
            if dummy_rb.isChecked():
                return False

        # You can add more validation checks for other responses if needed

        return True

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
