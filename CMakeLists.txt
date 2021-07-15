## ---------------------------------------------------------------------
##
## Copyright (c) 2020 - 2020 by the IBAMR developers
## All rights reserved.
##
## This file is part of IBAMR.
##
## IBAMR is free software and is distributed under the 3-clause BSD
## license. The full text of the license can be found in the file
## COPYRIGHT at the top level directory of IBAMR.
##
## ---------------------------------------------------------------------
CMAKE_MINIMUM_REQUIRED(VERSION 3.15.0)

PROJECT(AV_2D LANGUAGES C CXX Fortran)
FIND_PACKAGE(CCAD REQUIRED)

SET(CXX_SRC
    CirculationModel.cpp
    FeedbackForcer.cpp
    QFcn.cpp
    VelocityBcCoefs.cpp 
    RBFReconstructCacheOS.cpp 
    RBFOneSidedReconstructions.cpp
    )
ADD_EXECUTABLE(main2d main.cpp)
TARGET_SOURCES(main2d PRIVATE ${CXX_SRC})
TARGET_LINK_LIBRARIES(main2d CCAD::CCAD2d)
TARGET_LINK_LIBRARIES(main2d IBAMR::IBAMR2d)
