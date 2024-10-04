cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SegAltReview)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/nki-radiology/SegmentationReview/tree/segaltreview")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Anna Zapaishchykova (AIM Lab, BWH), Dr. Benjamin H. Kann (AIM Lab, BWH)")
set(EXTENSION_DESCRIPTION "Modified version of SegmentationReview by Anna Zapaishchykova (AIM Lab, BWH), Dr. Benjamin H. Kann (AIM Lab, BWH) to create batches of images to be segmented.")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SegAltReview)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
