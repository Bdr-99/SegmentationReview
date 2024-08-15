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
        self.parent.title = "SegmentationReview"
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
        self.dummy_radio_buttons = []

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
        self.atlasDirectoryButton.directoryChanged.connect(self.onAtlasDirectoryChanged)
        self.ui.save_and_next.connect('clicked(bool)', self.save_and_next_clicked)
        self.ui.overwrite_mask.connect('clicked(bool)', self.overwrite_mask_clicked)
        self.ui.next_case.connect('clicked(bool)', self.next_case_clicked)
        self.ui.prev_case.connect('clicked(bool)', self.prev_case_clicked)
        self.ui.delete_scan.connect('clicked(bool)', self.delete_scan_clicked)
        self.ui.btnToggleSegmentationDisplay.clicked.connect(self.toggleSegmentationDisplay)

        self.dummy_radio_buttons = [
            self.ui.radioButton_Dummy,  # for the generic likert score group
            self.ui.radioButton_PleuralEffusion_Dummy,  # for the PleuralEffusion group
            self.ui.radioButton_Atelectasis_Dummy,  # for the ChestWallMetastasis group
            self.ui.checkBox_ExtraThoracicMPM_Dummy,  # for the generic likert score group
            self.ui.radioButton_ChestWallMetastasis_Dummy,  # for the ChestWallMetastasis group
            self.ui.radioButton_Contrast_Dummy,  # for the PleuralEffusion group
            self.ui.radioButton_Confidence_Dummy
        ]

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

    def overwrite_mask_clicked(self):
        # overwrite self.segmentEditorWidget.segmentationNode()
        # segmentation_node = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLSegmentationNode')

        # Get the file path where you want to save the segmentation node
        file_path = self.directory + "/t.seg.nrrd"
        # Save the segmentation node to file as nifti
        i = 1  ## version number seg
        file_path_nifti = self.directory + "/" + \
                          self.segmentation_files[self.current_index].split("/")[-1].split(".nii.gz")[0] + "_v" + str(
            i) + ".nii.gz"
        # Save the segmentation node to file
        slicer.util.saveNode(self.segmentation_node, file_path)

        img = sitk.ReadImage(file_path)

        while os.path.exists(file_path_nifti):
            i += 1
            file_path_nifti = self.directory + "/" + \
                              self.segmentation_files[self.current_index].split("/")[-1].split(".nii.gz")[
                                  0] + "_v" + str(i) + ".nii.gz"
        print('Saving segmentation to file: ', file_path_nifti)
        sitk.WriteImage(img, file_path_nifti)

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

        # load the .csv file with the old annotations or create a new one
        if os.path.exists(directory + "/annotations.csv"):
            self.current_df = pd.read_csv(directory + "/annotations.csv")
            self.current_index = self.current_df.shape[0] + 1
            print("Restored current index: ", self.current_index)
        else:
            columns = [
                'file', 'generic_annotation', 'pleural_effusion',
                'atelectasis', 'extrathoracic', 'chest_wall_mets', 'contrast', 'confidence', 'comment'
            ]
            self.current_df = pd.DataFrame(columns=columns)
            self.current_index = 0

        # count the number of files in the directory
        for file in os.listdir(directory):
            if ".nii" in file and "_0000" in file:
                self.n_files += 1
                if os.path.exists(directory + "/" + file.split("_0000.nii.gz")[0] + ".nii.gz"):
                    self.nifti_files.append(directory + "/" + file)
                    self.segmentation_files.append(directory + "/" + file.split("_0000.nii.gz")[0] + ".nii.gz")
                else:
                    print("No mask for file: ", file)

        self.ui.status_checked.setText("Checked: " + str(self.current_index) + " / " + str(self.n_files - 1))
        self.resetUIElements()

        # load first file with mask
        self.load_nifti_file()

        # self.setup_and_load_scans_side_by_side(self.nifti_files, self.segmentation_files)
        self.time_start = time.time()

    def save_and_next_clicked(self):
        # Generic category (assuming it's kept the same for reference)

        if not self.all_responses_provided():
            print("Please provide all required responses before proceeding.")
            return

        generic_likert_score = self.get_likert_score_from_ui([
            self.ui.radioButton_1,
            self.ui.radioButton_2,
            self.ui.radioButton_3,
            self.ui.radioButton_4,
            self.ui.radioButton_5
        ])

        confidence_score = self.get_likert_score_from_ui([
            self.ui.radioButton_Confidence_1,
            self.ui.radioButton_Confidence_2,
            self.ui.radioButton_Confidence_3,
            self.ui.radioButton_Confidence_4,
            self.ui.radioButton_Confidence_5
        ])

        # Pleural effusion
        pleural_effusion_score = self.get_likert_score_from_ui([
            self.ui.radioButton_PleuralEffusion_1,
            self.ui.radioButton_PleuralEffusion_2,
            self.ui.radioButton_PleuralEffusion_3,
            self.ui.radioButton_PleuralEffusion_4,
            self.ui.radioButton_PleuralEffusion_5
        ])

        atelectasis_score = self.get_likert_score_from_ui([
            self.ui.checkBox_Atelectasis_No,
            self.ui.checkBox_Atelectasis_Yes,
        ])

        extrathoracic_score = self.get_likert_score_from_ui([
            self.ui.checkBox_ExtraThoracicMPM_No,
            self.ui.checkBox_ExtraThoracicMPM_Yes,
        ])

        chest_wall_mets_score = self.get_likert_score_from_ui([
            self.ui.radioButton_ChestWallMetastasis_No,
            self.ui.radioButton_ChestWallMetastasis_DrainSite,
            self.ui.radioButton_ChestWallMetastasis_Yes,
        ])

        contrast_score = self.get_likert_score_from_ui([
            self.ui.radioButton_Contrast_No,
            self.ui.radioButton_Contrast_Arterial,
            self.ui.radioButton_Contrast_Hepatic,
        ])
        # Now save them as before, but now with additional columns in your dataframe
        new_row = {
            'file': self.nifti_files[self.current_index].split(os.sep)[-1],
            'time_start': self.time_start,
            'time_end': time.time(),
            'time_elapsed': time.time() - self.time_start,
            'generic_annotation': generic_likert_score,
            'pleural_effusion': pleural_effusion_score,
            'atelectasis': atelectasis_score,
            'extrathoracic': extrathoracic_score,
            'chest_wall_mets': chest_wall_mets_score,
            'contrast': contrast_score,
            'confidence': confidence_score,
            'comment': self.ui.comment.toPlainText()
        }

        # Ensure self.current_df is a DataFrame
        if not isinstance(self.current_df, pd.DataFrame):
            print("Error: self.current_df is not a DataFrame!")
            return

        df = pd.DataFrame([new_row])
        df.to_csv(self.directory+"/annotations.csv", mode='a', index=False, header=False)

        self.overwrite_mask_clicked()
        if self.current_index < self.n_files - 1:
            self.current_index += 1
            self.load_nifti_file()
            self.time_start = time.time()
            self.ui.status_checked.setText("Checked: " + str(self.current_index) + " / " + str(self.n_files - 1))
            self.resetUIElements()
            self.ui.comment.setPlainText("")
        else:
            print("All files checked")

    def next_case_clicked(self):
        if self.current_index < self.n_files - 1:
            self.current_index += 1
            self.load_nifti_file()
            self.time_start = time.time()
            self.ui.status_checked.setText("Checked: " + str(self.current_index) + " / " + str(self.n_files - 1))
            self.resetUIElements()
            self.ui.comment.setPlainText("")


    def prev_case_clicked(self):
        if self.current_index != 0:
            self.current_index -= 1
            self.load_nifti_file()
            self.time_start = time.time()
            self.ui.status_checked.setText("Checked: " + str(self.current_index) + " / " + str(self.n_files - 1))
            self.resetUIElements()
            self.ui.comment.setPlainText("")


    def delete_scan_clicked(self):
        ## Delete the current scan from directory
        os.remove(self.nifti_files[self.current_index])
        ## Delete the current segmentation from the list
        os.remove(self.nifti_files[self.current_index].replace("_0000.nii.gz", ".nii.gz"))
        self.next_case_clicked()


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

        # ToDo: add 3d tumor view
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
        segmentationDisplayNode.SetVisibility2DFill(False)  # Do not show filled region in 2D
        segmentationDisplayNode.SetVisibility2DOutline(True)  # Show outline in 2D
        segmentationDisplayNode.SetColor(self.segmentation_color)
        segmentationDisplayNode.SetVisibility(True)

        self.set_segmentation_and_mask_for_segmentation_editor()

        print(file_path, segmentation_file_path)

    def setup_and_load_scans_side_by_side(self, volumePaths, segmentationPaths):
        # Ensure the current index is within bounds
        if self.current_index >= len(volumePaths) - 1:
            print("Reached the end or insufficient scans for side-by-side view.")
            return

        # Set the layout to show two views side by side
        layoutManager = slicer.app.layoutManager()
        layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutSideBySideView)

        # Load the current scan and segmentation
        volumeNodeCurrent = slicer.util.loadVolume(volumePaths[self.current_index])
        segmentationNodeCurrent = slicer.util.loadSegmentation(segmentationPaths[self.current_index])

        # Load the next scan and segmentation
        volumeNodeNext = slicer.util.loadVolume(volumePaths[self.current_index + 1])
        segmentationNodeNext = slicer.util.loadSegmentation(segmentationPaths[self.current_index + 1])

        # Assign the current scan to the first pane (Red)
        redCompositeNode = slicer.util.getNode('vtkMRMLSliceCompositeNodeRed')
        redCompositeNode.SetBackgroundVolumeID(volumeNodeCurrent.GetID())

        # Assign the next scan to the second pane (Yellow)
        yellowCompositeNode = slicer.util.getNode('vtkMRMLSliceCompositeNodeYellow')
        yellowCompositeNode.SetBackgroundVolumeID(volumeNodeNext.GetID())

        yellowSliceNode = slicer.util.getNode('vtkMRMLSliceNodeYellow')
        yellowSliceNode.SetOrientation("Axial")

        # Optionally, make the segmentations visible in their respective views
        # This can be done by adjusting the segmentation display settings as needed
        displayNodeCurrent = segmentationNodeCurrent.GetDisplayNode()
        displayNodeCurrent.SetVisibility(True)

        displayNodeNext = segmentationNodeNext.GetDisplayNode()
        displayNodeNext.SetVisibility(True)
        # Update the layout
        layoutManager.resetSliceViews()

    def resetUIElements(self):
        # Check all dummy radio buttons to effectively uncheck the other buttons in the group
        for dummy_rb in self.dummy_radio_buttons:
            dummy_rb.setChecked(True)

        # Reset the comment section
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
        for dummy_rb in self.dummy_radio_buttons:
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
