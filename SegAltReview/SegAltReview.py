import logging
import os


import vtk
from pathlib import Path
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import ctk
import qt
import json

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
# SegAltReview
#

class SegAltReview(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "SegAltReview"
        self.parent.categories = ["Examples"]  
        self.parent.dependencies = []  
        self.parent.contributors = ["Anna Zapaishchykova (BWH), Dr. Benjamin H. Kann, AIM-Harvard"]  
        self.parent.helpText = """
Modified version of SegmentationReview by Anna Zapaishchykova (AIM Lab, BWH), Dr. Benjamin H. Kann (AIM Lab, BWH) to create batches of images to be segmented.
"""
       
        self.parent.acknowledgementText = """
This file was developed by Anna Zapaishchykova, BWH, and adapted later. 
"""


#
# SegAltReviewWidget
#

class SegAltReviewWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        self.updated_segmentations = []
        self.directory = None
        self.current_index = 0
        self.likert_scores = []
        self.n_files = 0
        self.current_df = None
        
        # Set up the default directory
        config_file_path = Path(__file__).parent / 'config.json'
        
        # Load the defaults from the config file
        if config_file_path.exists():
            with config_file_path.open('r') as config_file:
                config = json.load(config_file)
                self.default_directory = Path(config.get("default_directory", ""))
                self.results_directory = Path(config.get("results_directory", ""))
                
        else:
            self.default_directory = None
            self.results_directory = None

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        # Setup the module widget
        ScriptedLoadableModuleWidget.setup(self)

        # Add a dropdown menu (combobox) below the directory widget
        self._createDropdownMenu()

        # Add directory input widget
        self._createDirectoryWidget()

        # Add custom UI widget
        self._createCustomUIWidget()

        # Add segment editor widget
        self._createSegmentEditorWidget()

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Connect GUI elements to respective functions
        self.atlasDirectoryButton.directoryChanged.connect(self.onAtlasDirectoryChanged)
        self.optionComboBox.currentIndexChanged.connect(self.onDropdownSelectionChanged)
        self.ui.save_next.connect('clicked(bool)', self.onSaveNextClicked)
        self.ui.previous.connect('clicked(bool)', self.onPreviousClicked)

    def _createDirectoryWidget(self):
        self.atlasDirectoryButton = ctk.ctkDirectoryButton()
        if self.default_directory:
            self.atlasDirectoryButton.directory = self.default_directory
            
        self.parametersFormLayout.addRow("Directory: ", self.atlasDirectoryButton)

    def _createDropdownMenu(self):
        """
        Creates a dropdown (combo box) for selecting an option.
        """
        # Add collapsible input section
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(parametersCollapsibleButton)

        # Add directory button to the input
        self.parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        # Create a label and dropdown (QComboBox)
        self.optionLabel = qt.QLabel("Radiologist:")
        self.optionComboBox = qt.QComboBox()

        # Add options to the dropdown menu
        self.optionComboBox.addItem("Please select your name")
        self.optionComboBox.addItem("Kalina")

        # Add the label and the dropdown to the layout underneath the directory widget
        self.parametersFormLayout.addRow(self.optionLabel, self.optionComboBox)

    def _createCustomUIWidget(self):
        # Load widget from .ui file (created by Qt Designer).
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/SegAltReview.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

    def _createSegmentEditorWidget(self):
        """Create and initialize a customize Slicer Editor which contains just some the tools that we need for the segmentation"""

        import qSlicerSegmentationsModuleWidgetsPythonQt

        # advancedCollapsibleButton
        self.segmentEditorWidget = qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget(
        )
        self.segmentEditorWidget.setMaximumNumberOfUndoStates(10)
        self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        self.segmentEditorWidget.unorderedEffectsVisible = False
        self.segmentEditorWidget.setEffectNameOrder([
            'Paint', 'Draw', 'Erase', 'Threshold', 'Smoothing',
        ])
        self.layout.addWidget(self.segmentEditorWidget)
        undoShortcut = qt.QShortcut(qt.QKeySequence('z'), self.segmentEditorWidget)
        undoShortcut.activated.connect(self.segmentEditorWidget.undo)

    def onAtlasDirectoryChanged(self, directory):
        try:
            if self.volume_node and slicer.mrmlScene.IsNodePresent(self.volume_node):
                slicer.mrmlScene.RemoveNode(self.volume_node)
            if self.segmentation_node and slicer.mrmlScene.IsNodePresent(self.segmentation_node):
                slicer.mrmlScene.RemoveNode(self.segmentation_node)
        except Exception as e:
            print(f"Error while removing nodes: {e}")

        # Clear the previously loaded image and segmentation
        if self.volume_node:
            slicer.mrmlScene.RemoveNode(self.volume_node)
        if self.segmentation_node:
            slicer.mrmlScene.RemoveNode(self.segmentation_node)

        # Set the new directory
        self.directory = Path(directory)
        self.parent_directory = self.directory.parent

        # Check if appropriate name has been chosen
        if not hasattr(self, 'batch_name') or self.batch_name == "Please select your name" :
            slicer.util.infoDisplay("Please select your name before starting.")
            return

        # Set the results_directory if not None
        if self.results_directory is None:
            self.results_directory = Path(self.directory / 'Results' / self.directory.name / self.batch_name)

        # Check if directory already exists
        if not self.batch_directory.exists():
            self.batch_directory.mkdir(parents=True, exist_ok=True)

        # Check if dataset contains the JSON file
        if Path(self.directory / 'file_paths.json').exists():
            with open(Path(self.directory / 'file_paths.json'), 'r') as file:
                self.file_paths = json.load(file)

        else:
            slicer.util.infoDisplay("Please create the appropriate JSON-file with the file paths.")
            return

        # Collect already annotated files or else create one
        if Path(self.batch_directory / 'SegAltReview_annotations.csv').exists():
            self.current_df = pd.read_csv(Path(self.batch_directory / 'SegAltReview_annotations.csv'), dtype=str)
            annotated_files = set(self.current_df['patientID'].values)

        else:
            self.current_df = pd.DataFrame(columns=['patientID', 'comment'])
            annotated_files = set()

        # Collect patients, images and segmentations in file paths JSON
        self.patientIDs = list(set(self.file_paths.keys()) - annotated_files)
        self.images = [Path(patientID / Path(self.file_paths[patientID]['image_folder'])) for patientID in self.patientIDs]
        #self.segmentations = [Path(self.file_paths[patientID]['segmentation_folder']) for patientID in self.patientIDs]

        # Set number of patients to check
        self.n_files = len(self.patientIDs)

        # Reset the UI to original
        self.resetUIElements()

        if self.n_files != 0:
            # Load the first case
            self.load_files()

        else:
            # Say that everything is already checked
            self.ui.var_check.setText("All files are checked!")
            self.ui.var_ID.setText('')
            print("All files checked")

    def onDropdownSelectionChanged(self):
        """
        Callback function for handling the dropdown selection.
        """

        # Set the batch name
        self.batch_name = self.optionComboBox.currentText

        # Check if appropriate name has been chosen
        if self.batch_name == "Please select your name":
            slicer.util.infoDisplay("Please select your name before starting.")
            return

        # Update the batch directory
        self.batch_directory = Path(self.results_directory / self.batch_name)

    def resetUIElements(self):
        self.ui.var_comment.clear()
        print("All UI elements reset.")

    def onSaveNextClicked(self):
        # Get the file path where you want to save the segmentation node
        seg_file_path = self.batch_directory / f'SegAltReview_{self.patientIDs[self.current_index]}.seg.nrrd'
                
        # Save the segmentation node to file
        slicer.util.saveNode(self.segmentation_node, str(seg_file_path))

        # Add to csv of annotations
        new_result = {
            'patientID': str(self.patientIDs[self.current_index]),
            'comment': self.ui.var_comment.toPlainText()
        }
        self.append_to_csv(new_result)
        
        # Add new segmentation path to updated list
        if str(seg_file_path) not in self.updated_segmentations:
            self.updated_segmentations.append(str(seg_file_path))
        
        # Go to next case
        self.goNext()

    def onPreviousClicked(self):
        # Return to previous case
        self.goPrevious()

    def goNext(self):
        try:
            if self.volume_node and slicer.mrmlScene.IsNodePresent(self.volume_node):
                slicer.mrmlScene.RemoveNode(self.volume_node)
            if self.segmentation_node and slicer.mrmlScene.IsNodePresent(self.segmentation_node):
                slicer.mrmlScene.RemoveNode(self.segmentation_node)
        except Exception as e:
            print(f"Error while removing nodes: {e}")

        if self.current_index < self.n_files - 1:
            self.current_index += 1
            self.load_files()
            self.resetUIElements()
        else:
            self.ui.var_check.setText("All files are checked!")
            self.ui.var_ID.setText('')
            print("All files checked")

    def goPrevious(self):
        if self.current_index > 0:
            try:
                if self.volume_node and slicer.mrmlScene.IsNodePresent(self.volume_node):
                    slicer.mrmlScene.RemoveNode(self.volume_node)
                if self.segmentation_node and slicer.mrmlScene.IsNodePresent(self.segmentation_node):
                    slicer.mrmlScene.RemoveNode(self.segmentation_node)
            except Exception as e:
                print(f"Error while removing nodes: {e}")

            self.current_index -= 1
            self.current_df = pd.read_csv(self.batch_directory / 'SegAltReview_annotations.csv', dtype=str)
            self.load_files()
            self.resetUIElements()

        else:
            print('Already at start of sequence!')
    
    def append_to_csv(self, new_row_data):
        # Define the required column order
        required_columns = ['patientID', 'comment']

        # Ensure all required columns are present in the new row, filling in None for any that are missing
        new_row = {column: str(new_row_data.get(column, None)) if new_row_data.get(column) is not None else None for column in required_columns}

        # Full path to the CSV file
        file_path = Path(self.batch_directory / 'SegAltReview_annotations.csv')
        
        # Check if the patientID already exists
        if new_row['patientID'] in self.current_df['patientID'].values:
            # Find the index of the existing row
            existing_row_index = self.current_df.index[self.current_df['patientID'] == new_row['patientID']].tolist()[0]

            # Merge the comments
            existing_comment = self.current_df.at[existing_row_index, 'comment']
            new_comment = new_row['comment']
            
            if pd.notna(existing_comment) and pd.notna(new_comment):
                # Concatenate comments with a separator (e.g., "; ")
                merged_comment = f"{existing_comment} | {new_comment}"
            elif pd.isna(existing_comment):
                merged_comment = new_comment
            else:
                merged_comment = existing_comment

            # Update the DataFrame with the merged comment            
            self.current_df.at[existing_row_index, 'comment'] = merged_comment
            
        else:
            # If patientID does not exist, append the new row using pd.concat
            new_row_df = pd.DataFrame([new_row])  # Convert the new row to a DataFrame
            self.current_df = pd.concat([self.current_df, new_row_df], ignore_index=True)

        # Write the updated DataFrame back to the CSV file
        self.current_df.to_csv(file_path, index=False)
        
    def load_files(self):
        # Load image
        image_path = Path(self.directory / self.images[self.current_index] / '1-001.dcm')
        self.volume_node = slicer.util.loadVolume(image_path)
        slicer.app.applicationLogic().PropagateVolumeSelection(0)

        # Retrieve segmentation path
        if not self.current_index < len(self.updated_segmentations):
            segmentation_file_path = None

        else:
            segmentation_file_path = self.updated_segmentations[self.current_index]

        if segmentation_file_path is not None:
            # Load segmentation
            self.segmentation_node = slicer.util.loadSegmentation(segmentation_file_path)
            self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

            # Harden any transformations applied to the segmentation or volume nodes
            slicer.vtkSlicerTransformLogic().hardenTransform(self.segmentation_node)
            slicer.vtkSlicerTransformLogic().hardenTransform(self.volume_node)

        else:
            # Create a new segmentation
            self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
            self.segmentation_node.SetName("Segmentation")

            # Add a display node to the segmentation node
            segmentationDisplayNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationDisplayNode')
            self.segmentation_node.SetAndObserveDisplayNodeID(segmentationDisplayNode.GetID())

            # Get the segmentation object from the node
            seg = self.segmentation_node.GetSegmentation()
            seg_id = seg.AddEmptySegment('Tumor')
            segment = seg.GetSegment(seg_id)
            segment.SetColor([1.0, 1.0, 0.0])

        # Setting the visualization of the segmentation to outline only
        segmentationDisplayNode = self.segmentation_node.GetDisplayNode()
        segmentationDisplayNode.SetVisibility2DFill(False)  # Do not show filled region in 2D
        segmentationDisplayNode.SetVisibility2DOutline(True)  # Show outline in 2D
        segmentationDisplayNode.SetVisibility(True)

        # Connect segmentation editor to the masks
        self.set_segmentation_and_mask_for_segmentation_editor()
        self.ui.var_check.setText(str(self.current_index) + " / " + str(self.n_files))
        self.ui.var_ID.setText(str(self.patientIDs[self.current_index]))

    def set_segmentation_and_mask_for_segmentation_editor(self):
        slicer.app.processEvents()
        self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        self.segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
        slicer.mrmlScene.AddNode(self.segmentEditorNode)
        self.segmentEditorWidget.setMRMLSegmentEditorNode(self.segmentEditorNode)
        self.segmentEditorWidget.setSegmentationNode(self.segmentation_node)
        self.segmentEditorWidget.setActiveEffectByName("Paint")
        self.segmentEditorWidget.setSourceVolumeNode(self.volume_node)
        self.segmentEditorNode.SetOverwriteMode(2)
        self.segmentEditorNode.SetMaskMode(4)
   




























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
        #self.initializeParameterNode()

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

        #if inputParameterNode:
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
