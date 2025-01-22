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
import re

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
        self.roi_display = None
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
        self.ui.bounding.connect('clicked(bool)', self.onToggleClicked)

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
        self.optionLabel = qt.QLabel("Rater:")
        self.optionComboBox = qt.QComboBox()

        # Add options to the dropdown menu
        self.optionComboBox.addItem("Please select your name")
        self.optionComboBox.addItem("Kalina")
        self.optionComboBox.addItem("Lily")
        self.optionComboBox.addItem("George")
        self.optionComboBox.addItem("Daan")

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
            # Get all nodes in the scene
            nodes = slicer.mrmlScene.GetNodes()
            node_count = nodes.GetNumberOfItems()
            
            # Iterate over all nodes and remove only data-related nodes
            for i in range(node_count):
                node = nodes.GetItemAsObject(0)  # Always get the first node, as the list updates when nodes are removed
                
                # Check if the node is a data node (e.g., volumes, segmentations, models, etc.)
                if node.IsA("vtkMRMLScalarVolumeNode") or \
                   node.IsA("vtkMRMLSegmentationNode") or \
                   node.IsA("vtkMRMLModelNode") or \
                   node.IsA("vtkMRMLMarkupsNode"):  # Add other data node types as needed
                    slicer.mrmlScene.RemoveNode(node)
            
            print("All data nodes removed successfully.")
        except Exception as e:
            print(f"Error while removing data nodes: {e}")

        # Set the new directory
        self.directory = Path(directory)
        self.parent_directory = self.directory.parent

        # Check if appropriate name has been chosen
        if not hasattr(self, 'batch_name') or self.batch_name == "Please select your name":
            if self.default_directory:
                self.atlasDirectoryButton.directory = self.default_directory
            
            slicer.util.infoDisplay("Please select your name before starting.")
            return

        # Set the results_directory if not None
        if self.results_directory is None:
            self.results_directory = Path(self.directory / 'Results' / self.directory.name)
            self.batch_directory = Path(self.results_directory / self.batch_name)
        
        else:
            self.results_directory = Path(self.results_directory / self.directory.name)
            self.batch_directory = Path(self.results_directory / self.batch_name)

        # Check if directory already exists
        if not self.batch_directory.exists():
            self.batch_directory.mkdir(parents=True, exist_ok=True)

        # Check if dataset contains the JSON file
        if Path(self.directory / 'radiomics_study.json').exists():
            with open(Path(self.directory / 'radiomics_study.json'), 'r') as file:
                self.file_paths = json.load(file)

        else:
            slicer.util.infoDisplay("Please create the appropriate JSON-file with the file paths.")
            return

        # Collect already annotated files or else create one
        if Path(self.batch_directory / 'SegAltReview_annotations.csv').exists():
            self.current_df = pd.read_csv(Path(self.batch_directory / 'SegAltReview_annotations.csv'), dtype=str)
            annotated_files = set(self.current_df['patientID'].values)
            if annotated_files:
                self.updated_segmentations = {patientID: f'SegAltReview_{patientID}.seg.nrrd' for patientID in sorted(annotated_files)}

        else:
            self.current_df = pd.DataFrame(columns=['patientID', 'comment'])
            annotated_files = set()

        # Collect patients, images and segmentations in file paths JSON
        self.patientIDs = sorted(list(self.file_paths.keys()))  # Ensure sorted patientIDs
        self.new_patientIDs = sorted(list(set(self.patientIDs) - annotated_files))  # Sort the difference as well
        self.old_patientIDs = sorted(list(set(self.patientIDs)- set(self.new_patientIDs)))

        # Update patientIDs to reflect old vs new
        self.patientIDs = self.old_patientIDs + self.new_patientIDs
        
        # Create dictionaries for images and segmentations
        self.images = {
        patientID: (
            Path(patientID) / Path(self.file_paths.get(patientID, {}).get('image_file'))
            if self.file_paths.get(patientID, {}).get('image_file') is not None
            else None
            )
        for patientID in self.patientIDs
        }

        self.original_segmentations = {
            patientID: (
                Path(patientID) / Path(self.file_paths.get(patientID, {}).get('segmentation_file'))
                if self.file_paths.get(patientID, {}).get('segmentation_file') is not None
                else None
            )
            for patientID in self.patientIDs
        }

        # Set number of patients
        self.n_files = len(self.patientIDs)
        
        # Set current index of patients to check
        self.current_index = int(self.n_files - len(self.new_patientIDs))
        print(self.patientIDs)
        print(self.current_index)
        
        
        # Reset the UI to original
        self.resetUIElements()

        if len(self.new_patientIDs) != 0:
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
        if self.patientIDs[self.current_index] not in self.updated_segmentations:
            self.updated_segmentations[self.patientIDs[self.current_index]] = (str(seg_file_path))
        
        # Go to next case
        self.goNext()

    def onPreviousClicked(self):
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
        
        # Return to previous case
        self.goPrevious()
        
    def onToggleClicked(self):
        if self.roi_display:
            # Get the current visibility state
            current_visibility = self.roi_display.GetVisibility()
            
            # Toggle visibility
            new_visibility = 0 if current_visibility else 1
            self.roi_display.SetVisibility(new_visibility)
            
    def goNext(self):
        try:
            if self.volume_node and slicer.mrmlScene.IsNodePresent(self.volume_node):
                slicer.mrmlScene.RemoveNode(self.volume_node)
            if self.segmentation_node and slicer.mrmlScene.IsNodePresent(self.segmentation_node):
                slicer.mrmlScene.RemoveNode(self.segmentation_node)
            if self.roi_node and slicer.mrmlScene.IsNodePresent(self.roi_node):
                slicer.mrmlScene.RemoveNode(self.roi_node)
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
                if self.roi_node and slicer.mrmlScene.IsNodePresent(self.roi_node):
                    slicer.mrmlScene.RemoveNode(self.roi_node)
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
        image_path = Path(self.directory / self.images[self.patientIDs[self.current_index]])
        self.volume_node = slicer.util.loadVolume(image_path)
        slicer.app.applicationLogic().PropagateVolumeSelection(0)

        # Retrieve segmentation path
        if not self.patientIDs[self.current_index] in self.updated_segmentations:
            segmentation_file_path = None

        else:
            segmentation_file_path = Path(self.batch_directory / self.updated_segmentations[self.patientIDs[self.current_index]])

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
        
        # Check if the current segmentation is not None
        if self.original_segmentations[self.patientIDs[self.current_index]] is not None:
            # Set bounding box
            extents = []
            space_origin = None
            space_directions = []

            # Read file header
            with open(Path(self.directory / self.original_segmentations[self.patientIDs[self.current_index]]), 'rb') as file:
                current_segment_name = None  # Track the name of the current segment
                temp_extent = None  # Temporarily store extent until matched with a name

                for line in file:
                    try:
                        line = line.decode('utf-8')
                        if line.strip() == "":
                            break

                        # Check for segment name
                        if line.startswith("Segment") and "_Name:=" in line:
                            current_segment_name = re.match(r"Segment\d+_Name:=([\w\s,]+)", line).group(1).strip()

                            # If extent is already found and name matches, add to extents
                            if current_segment_name == "Neoplasm, Primary" and temp_extent is not None:
                                extents.append(temp_extent)
                                temp_extent = None  # Clear temp_extent after appending
                                current_segment_name = None
                                
                        # Check for extent
                        elif line.startswith("Segment") and "_Extent:=" in line:
                            temp_extent = list(map(int, re.match(r"Segment\d+_Extent:=([\d\s]+)", line).group(1).split()))

                            # If name is already found and matches, add to extents
                            if current_segment_name == "Neoplasm, Primary":
                                extents.append(temp_extent)
                                temp_extent = None  # Clear temp_extent after appending
                                current_segment_name = None

                        # Handle space origin
                        elif line.startswith("space origin:"):
                            space_origin = np.array(list(map(float, re.findall(r"-?\d+\.\d+|-?\d+", line))))

                        # Handle space directions
                        elif line.startswith("space directions:"):
                            directions = re.findall(r'\(.*?\)|none', line)
                            for dir_line in directions:
                                if dir_line != "none":
                                    # Extract numbers and convert to float (works for both integers and floats)
                                    space_directions.append([float(num) for num in re.findall(r"-?\d+\.\d+|-?\d+", dir_line)])

                    except UnicodeDecodeError:
                        break
            
            
            # Prepare data
            extents = np.array(extents)
            voxel_min = np.min(extents[:, [0, 2, 4]], axis=0)
            voxel_max = np.max(extents[:, [1, 3, 5]], axis=0)

            # Convert voxel extents to spatial coordinates
            space_directions = np.array(space_directions[-3:])  # Keep last 3 directions
            spatial_min = space_origin + np.dot(space_directions, voxel_min)
            spatial_max = space_origin + np.dot(space_directions, voxel_max)

            # Adjust for RAS system: negate X and Y coordinates
            spatial_min[:2] = -spatial_min[:2]  # Negate L-R and P-A (X and Y)
            spatial_max[:2] = -spatial_max[:2]  # Negate L-R and P-A (X and Y)

            # Compute spatial size and apply margin
            spatial_size = spatial_max - spatial_min
            margin = 0.5 * spatial_size  # 30% margin in spatial coordinates
            spatial_min = spatial_min - margin / 2  # Expand equally on both sides
            spatial_max = spatial_max + margin / 2
            spatial_center = (spatial_min + spatial_max) / 2
            spatial_size = spatial_max - spatial_min

            # Create ROI node
            self.roi_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "BoundingBox")
            self.roi_node.SetCenter(spatial_center.tolist())
            self.roi_node.SetSize(spatial_size.tolist())
            self.roi_node.SetNthControlPointVisibility(0, False)
            
            # Style the bounding box
            self.roi_display = self.roi_node.GetDisplayNode()
            self.roi_display.SetVisibility(True)  # Make the ROI visible
            self.roi_display.SetHandlesInteractive(False)  # Disable interaction handles
            self.roi_display.SetPointLabelsVisibility(False)  # Disable point labels
            self.roi_display.SetFillVisibility(False)  # Disable the fill
            self.roi_display.SetPropertiesLabelVisibility(False)
            self.roi_display.SetSelectedColor(1.0, 0.0, 0.0)
            self.roi_display.SetActiveColor(1.0, 0.0, 0.0)
            self.roi_display.SetColor(1.0, 0.0, 0.0)  # Set bounding box color (red)


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
