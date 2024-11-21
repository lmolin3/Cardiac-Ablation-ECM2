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

  set(ELECTROSTATICS_DIR ${CMAKE_SOURCE_DIR}/miniapps/electrostatics-ecm2)
  set(ELECTROSTATICS_EXTRA_INCLUDE_DIRECTORIES
    ${ELECTROSTATICS_DIR}
    ${CMAKE_SOURCE_DIR}/miniapps/common-ecm2
    ${CMAKE_SOURCE_DIR}/miniapps/common)

  add_library(electrostatics_ecm2
    ${ELECTROSTATICS_DIR}/lib/electrostatics_solver.cpp
    ${ELECTROSTATICS_DIR}/bc/electrostatics_bchandler.cpp
    ${CMAKE_SOURCE_DIR}/miniapps/common-ecm2/utils.cpp
    ${CMAKE_SOURCE_DIR}/miniapps/common/mesh_extras.cpp
    ${CMAKE_SOURCE_DIR}/miniapps/common/fem_extras.cpp
    ${CMAKE_SOURCE_DIR}/miniapps/common/pfem_extras.cpp)

  target_include_directories(electrostatics_ecm2 PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${MFEM_INCLUDE_DIRS}
    ${TPL_INCLUDE_DIRS}
    ${ELECTROSTATICS_EXTRA_INCLUDE_DIRECTORIES})

    
  target_link_libraries(electrostatics_ecm2
    mfem
    ${TPL_LIBRARIES})

  add_mfem_miniapp(electrostatics_test2D
    MAIN tests/electrostatics_test2D.cpp
    ${ELECTROSTATICS_COMMON_FILES}
    LIBRARIES mfem mfem-common electrostatics_ecm2)

  target_include_directories(electrostatics_test2D
    PRIVATE ${ELECTROSTATICS_EXTRA_INCLUDE_DIRECTORIES})


  add_mfem_miniapp(electrostatics_test3D
    MAIN tests/electrostatics_test3D.cpp
    ${ELECTROSTATICS_COMMON_FILES}
    LIBRARIES mfem mfem-common electrostatics_ecm2)

  target_include_directories(electrostatics_test3D
    PRIVATE ${ELECTROSTATICS_EXTRA_INCLUDE_DIRECTORIES})


    add_mfem_miniapp(electrostatics_test3D_aniso
    MAIN tests/electrostatics_test3D_anisotropic.cpp
    ${ELECTROSTATICS_COMMON_FILES}
    LIBRARIES mfem mfem-common electrostatics_ecm2)

  target_include_directories(electrostatics_test3D_aniso
    PRIVATE ${ELECTROSTATICS_EXTRA_INCLUDE_DIRECTORIES})


    add_mfem_miniapp(electrostatics_test3D_aniso_rotated
    MAIN tests/electrostatics_test3D_anisotropic_rotated.cpp
    ${ELECTROSTATICS_COMMON_FILES}
    LIBRARIES mfem mfem-common electrostatics_ecm2)

  target_include_directories(electrostatics_test3D_aniso_rotated
    PRIVATE ${ELECTROSTATICS_EXTRA_INCLUDE_DIRECTORIES})


  add_mfem_miniapp(mms_2D
    MAIN tests/mms_2D.cpp
    ${ELECTROSTATICS_COMMON_FILES}
    LIBRARIES mfem mfem-common electrostatics_ecm2)

  target_include_directories(mms_2D
    PRIVATE ${ELECTROSTATICS_EXTRA_INCLUDE_DIRECTORIES})

  add_mfem_miniapp(convergence_2D
    MAIN tests/convergence_2D.cpp
    ${ELECTROSTATICS_COMMON_FILES}
    LIBRARIES mfem mfem-common electrostatics_ecm2)

  target_include_directories(convergence_2D
    PRIVATE ${ELECTROSTATICS_EXTRA_INCLUDE_DIRECTORIES})
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

