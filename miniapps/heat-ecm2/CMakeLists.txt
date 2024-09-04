# Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

if (MFEM_USE_MPI)
  add_subdirectory(lib)

  set(HEAT_DIR ${CMAKE_SOURCE_DIR}/miniapps/heat-ecm2)
  set(HEAT_EXTRA_INCLUDE_DIRECTORIES
    ${HEAT_DIR}
    ${CMAKE_SOURCE_DIR}/miniapps/common-ecm2
    ${CMAKE_SOURCE_DIR}/miniapps/common)

  add_library(heat_ecm2
    ${HEAT_DIR}/lib/heat_solver.cpp
    ${HEAT_DIR}/lib/heat_operator.cpp
    ${HEAT_DIR}/bc/heat_bchandler.cpp
    ${CMAKE_SOURCE_DIR}/miniapps/common-ecm2/utils.cpp
    ${CMAKE_SOURCE_DIR}/miniapps/common/mesh_extras.cpp
    ${CMAKE_SOURCE_DIR}/miniapps/common/fem_extras.cpp
    ${CMAKE_SOURCE_DIR}/miniapps/common/pfem_extras.cpp)

  target_include_directories(heat_ecm2 PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${MFEM_INCLUDE_DIRS}
    ${TPL_INCLUDE_DIRS}
    ${HEAT_EXTRA_INCLUDE_DIRECTORIES})

    
  target_link_libraries(heat_ecm2
    mfem
    ${TPL_LIBRARIES})


add_mfem_miniapp(heat_test
    MAIN tests/heat_test.cpp
    ${HEAT_COMMON_FILES}
    LIBRARIES mfem mfem-common heat_ecm2)

  target_include_directories(heat_test
    PRIVATE ${HEAT_EXTRA_INCLUDE_DIRECTORIES}) 

add_mfem_miniapp(convergence
    MAIN tests/convergence.cpp
    ${HEAT_COMMON_FILES}
    LIBRARIES mfem mfem-common heat_ecm2)

  target_include_directories(convergence
    PRIVATE ${HEAT_EXTRA_INCLUDE_DIRECTORIES}) 

  add_mfem_miniapp(heat_test_3Daniso
    MAIN tests/heat_test_3Daniso.cpp
    ${HEAT_COMMON_FILES}
    LIBRARIES mfem mfem-common heat_ecm2)

  target_include_directories(heat_test_3Daniso
    PRIVATE ${HEAT_EXTRA_INCLUDE_DIRECTORIES}) 

  #add_mfem_miniapp(HEAT_test3D
   # MAIN tests/HEAT_test3D.cpp
   # ${HEAT_COMMON_FILES}
   # LIBRARIES mfem mfem-common heat_ecm2)

  #target_include_directories(HEAT_test3D
  #  PRIVATE ${HEAT_EXTRA_INCLUDE_DIRECTORIES})


  #  add_mfem_miniapp(HEAT_test3D_aniso
  #  MAIN tests/HEAT_test3D_anisotropic.cpp
  #  ${HEAT_COMMON_FILES}
  #  LIBRARIES mfem mfem-common heat_ecm2)

  #target_include_directories(HEAT_test3D_aniso
  #  PRIVATE ${HEAT_EXTRA_INCLUDE_DIRECTORIES})


  #  add_mfem_miniapp(HEAT_test3D_aniso_rotated
  #  MAIN tests/HEAT_test3D_anisotropic_rotated.cpp
  #  ${HEAT_COMMON_FILES}
  #  LIBRARIES mfem mfem-common heat_ecm2)

  #target_include_directories(HEAT_test3D_aniso_rotated
  #  PRIVATE ${HEAT_EXTRA_INCLUDE_DIRECTORIES})

endif () #MFEM_USE_MPI

#  if (MFEM_ENABLE_TESTING)
#    foreach (test "mms" )
#      add_test(NAME navier_${test}_np${MFEM_MPI_NP}
#        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
#        ${MPIEXEC_PREFLAGS}
#        $<TARGET_FILE:navier_${test}> -cr -no-vis
#        ${MPIEXEC_POSTFLAGS})
#    endforeach()
# endif() #MFEM_ENABLE_TESTING

