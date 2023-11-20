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


#
# SegmentationReview
#

class SegmentationReview(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "SegmentationReview - NET"
        self.parent.categories = ["Segmentation"]
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
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/SegmentationReview.ui'))

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
        defaultVolumeDisplayNode.SetWindowLevelLocked(True)
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

    def onAtlasDirectoryChanged(self, directory):
        if self.volume_node:
            slicer.mrmlScene.RemoveNode(self.volume_node)
        if self.segmentation_node:
            slicer.mrmlScene.RemoveNode(self.segmentation_node)

        self.directory = directory

        # Initialize these variables at the beginning
        self.n_files = 0
        self.nifti_files = []
        self.segmentation_files = []

        # collect images and masks
        for seg_file in glob.glob(directory + '/*.seg.nii.gz'):
            self.n_files += 1
            file = seg_file.replace(".seg", "")
            self.nifti_files.append(file)
            self.segmentation_files.append(seg_file)

        # load the .csv file with the old annotations or create a new one
        if os.path.exists(directory + "/NET_annotations.csv"):
            self.current_df = pd.read_csv(directory + "/NET_annotations.csv")
            files_set = set(self.current_df['files'].values)
            self.nifti_files = [file for file in self.nifti_files if file not in files_set]
            self.current_index = self.current_df.shape[0] + 1
            print("Restored current index: ", self.current_index)
        else:
            columns = [
                'file', 'is_liver_imaged', 'contrast', 'phase_timing', 
                'ai_segmentation_quality', 'manual_segmentation_confidence', 
                'adjustment_segmentation_difficulty', 'high_signal_to_noise', 
                'metal_artifacts', 'patient_motion', 'other', 'time-stamp'
            ]
            self.current_df = pd.DataFrame(columns=columns)
            self.current_index = 0

        # count the number of files in the directory
        self.ui.status_checked.setText("Checked: " + str(self.current_index) + " / " + str(self.n_files))
        self.resetUIElements()

        # load first file with mask
        self.load_nifti_file()
        self.time_start = time.time()

    def onNothingToSegmentClicked(self):

        new_row = {
            'file': self.nifti_files[self.current_index].split(os.sep)[-1],
            'is_liver_imaged': None,
            'contrast': None,
            'phase_timing': None,
            'ai_segmentation_quality': None,
            'manual_segmentation_confidence': None,
            'adjustment_segmentation_difficulty': None,
            'high_signal_to_noise': None,
            'metal_artifacts': None,
            'patient_motion': None,
            'other': None,
            'time-stamp': time.time()
        }

        # Ensure self.current_df is a DataFrame
        if not isinstance(self.current_df, pd.DataFrame):
            print("Error: self.current_df is not a DataFrame!")
            return 

        df = pd.DataFrame([new_row])
        df.to_csv(self.directory+"/NET_annotations.csv", mode='a', index=False, header=False)

        # self.overwrite_mask_clicked()
        if self.current_index < self.n_files - 1:
            self.current_index += 1
            self.load_nifti_file()
            self.time_start = time.time()
            self.ui.status_checked.setText("Checked: " + str(self.current_index) + " / " + str(self.n_files))
            self.resetUIElements()
            self.ui.comment.setPlainText("")
        else:
            self.ui.status_checked.setText("All files are checked!!")
            print("All files checked")


    def onNoLiverToSegmentClicked(self):

        new_row = {
            'file': self.nifti_files[self.current_index].split(os.sep)[-1],
            'is_liver_imaged': 'Not at all',
            'contrast': None,
            'phase_timing': None,
            'ai_segmentation_quality': None,
            'manual_segmentation_confidence': None,
            'adjustment_segmentation_difficulty': None,
            'high_signal_to_noise': None,
            'metal_artifacts': None,
            'patient_motion': None,
            'other': None,
            'time-stamp': time.time()
        }

        # Ensure self.current_df is a DataFrame
        if not isinstance(self.current_df, pd.DataFrame):
            print("Error: self.current_df is not a DataFrame!")
            return 

        df = pd.DataFrame([new_row])
        df.to_csv(self.directory+"/NET_annotations.csv", mode='a', index=False, header=False)

        # self.overwrite_mask_clicked()
        if self.current_index < self.n_files - 1:
            self.current_index += 1
            self.load_nifti_file()
            self.time_start = time.time()
            self.ui.status_checked.setText("Checked: " + str(self.current_index) + " / " + str(self.n_files))
            self.resetUIElements()
            self.ui.comment.setPlainText("")
        else:
            self.ui.status_checked.setText("All files are checked!!")
            print("All files checked")


    def onSaveAndNextClicked(self):
        # Generic category (assuming it's kept the same for reference)

        if not self.all_responses_provided():
            print("Please provide all required responses before proceeding.")
            return

        # Collect responses from the UI
        liver_imaged_score = self.get_likert_score_from_ui([
            self.ui.radioLiverYes,
            self.ui.radioLiverNo
        ])

        contrast_score = self.get_likert_score_from_ui([
            self.ui.radioButtonContrastNone,
            self.ui.radioButtonContrastArterial,
            self.ui.radioButtonContrastPortal,
            self.ui.radioButtonContrastLatePhase
        ])

        phase_timing_score = self.get_likert_score_from_ui([
            self.ui.radioButtonPhaseTooEarly,
            self.ui.radioButtonPhaseJustRight,
            self.ui.radioButtonPhaseTooLate
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

        # Ensure self.current_df is a DataFrame
        if not isinstance(self.current_df, pd.DataFrame):
            print("Error: self.current_df is not a DataFrame!")
            return 

        df = pd.DataFrame([new_row])
        df.to_csv(self.directory+"/NET_annotations.csv", mode='a', index=False, header=False)

        # self.overwrite_mask_clicked()
        if self.current_index <= self.n_files:
            self.current_index += 1
            self.load_nifti_file()
            self.time_start = time.time()
            self.ui.status_checked.setText("Checked: " + str(self.current_index) + " / " + str(self.n_files))
            self.resetUIElements()
            self.ui.comment.setPlainText("")
        else:
            self.ui.status_checked.setText("All files are checked!!")
            print("All files checked")

    def onNextCaseClicked(self):
        if self.current_index <= self.n_files:
            self.current_index += 1
            self.load_nifti_file()
            self.time_start = time.time()
            self.ui.status_checked.setText("Checked: " + str(self.current_index) + " / " + str(self.n_files))
            self.resetUIElements()
            self.ui.comment.setPlainText("")

    def onPreviousCaseClicked(self):
        if self.current_index != 0:
            self.current_index -= 1
            self.load_nifti_file()
            self.time_start = time.time()
            self.ui.status_checked.setText("Checked: " + str(self.current_index) + " / " + str(self.n_files))
            self.resetUIElements()
            self.ui.comment.setPlainText("")

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

        print(file_path, segmentation_file_path)

    def resetUIElements(self):
        # Check all dummy radio buttons to effectively uncheck the other buttons in the group
        self.ui.radioLiverHidden.setChecked(True)
        self.ui.radioButtonContrastHidden.setChecked(True)
        self.ui.radioButtonPhaseTimingHidden.setChecked(True)
        self.ui.radioButtonAISegmentationHidden.setChecked(True)
        self.ui.radioButtonManualSegmentationHidden.setChecked(True)
        self.ui.radioButtonDifficultySegmentationHidden.setChecked(True)
        self.ui.checkNoiseRatio.setChecked(False)
        self.ui.checkMetalArtifacts.setChecked(False)
        self.ui.checkPatientMotion.setChecked(False)
        self.ui.checkOtherReasons.setChecked(False)
        self.ui.lineEditOtherReasons.setText("")
        self.ui.comment.setPlainText("")
        print("All UI elements reset.")

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
