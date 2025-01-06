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

  set(NAVIER_DIR ${CMAKE_SOURCE_DIR}/miniapps/navier-ecm2)

  include_directories(
    ${CMAKE_SOURCE_DIR}/miniapps/common
    ${CMAKE_SOURCE_DIR}/miniapps/common-ecm2
    ${NAVIER_DIR}
  )

    list(APPEND NAVIER_SRCS
    lib/navier_solver.cpp
    lib/navier_preconditioners.cpp
    bc/navier_bchandler.cpp
    integrators/custom_integrators.cpp
  )

  list(APPEND NAVIER_HDRS
    lib/navier_solver.hpp
    lib/navier_preconditioners.hpp
    lib/navier_qoi.hpp
    bc/navier_bchandler.hpp
    integrators/custom_integrators.hpp
  )

  mfem_add_library(navier_ct_ecm2 ${NAVIER_SRCS} ${NAVIER_HDRS})

    
  target_link_libraries(navier_ct_ecm2
    mfem
    mfem-common
    mfem-common-ecm2
    ${TPL_LIBRARIES})

  add_mfem_miniapp(navier-poiseulle
    MAIN tests/navier_poiseulle.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common navier_ct_ecm2)

  add_mfem_miniapp(navier-mms
    MAIN tests/navier_mms.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common navier_ct_ecm2)

  add_mfem_miniapp(navier-flowaroundcyl
    MAIN tests/navier_flowaroundcyl.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common navier_ct_ecm2)

  add_mfem_miniapp(navier-liddriven
    MAIN tests/navier_liddriven.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common navier_ct_ecm2)

  add_mfem_miniapp(navier-backstep
    MAIN tests/navier_backstep.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common navier_ct_ecm2)

  add_custom_command(TARGET navier-poiseulle POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/Mesh Mesh
            COMMENT "Copying the navier-ecm2 Mesh data directory ...")

  #target_include_directories(navier_test2D
  #  PRIVATE ${NAVIER_EXTRA_INCLUDE_DIRECTORIES})


endif () #MFEM_USE_MPI
