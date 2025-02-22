set(MESH_FILES
  multidomain-hex.mesh
)

# Add a target to copy the mesh files from the source directory; used by sample
# runs.
set(SRC_MESH_FILES)
foreach(MESH_FILE ${MESH_FILES})
  list(APPEND SRC_MESH_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${MESH_FILE})
endforeach()
add_custom_command(OUTPUT data_is_copied
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC_MESH_FILES} .
  COMMAND ${CMAKE_COMMAND} -E touch data_is_copied
  COMMENT "Copying multidomain miniapps data files ...")
add_custom_target(copy_miniapps_multidomain_test_data DEPENDS data_is_copied)


# Add the include directories
include_directories(
  ${CMAKE_SOURCE_DIR}/miniapps/common
  ${CMAKE_SOURCE_DIR}/miniapps/common-ecm2
  ${CMAKE_SOURCE_DIR}/miniapps/heat-ecm2
  ${CMAKE_SOURCE_DIR}/miniapps/celldeath-ecm2
  ${CMAKE_SOURCE_DIR}/miniapps/electrostatics-ecm2
  ${CMAKE_SOURCE_DIR}/miniapps/navier-monolithic-ecm2
)

# Parallel apps.
if (MFEM_USE_MPI)
  add_mfem_miniapp(multidomain-dirichlet
    MAIN multidomain-dirichlet.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common mfem-common-ecm2 heat-ecm2)
  add_dependencies(multidomain-dirichlet copy_miniapps_multidomain_test_data)


  add_mfem_miniapp(multidomain-three-domains-celldeath
    MAIN multidomain-three-domains-celldeath.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common mfem-common-ecm2 heat-ecm2 celldeath-ecm2)
  add_dependencies(multidomain-three-domains-celldeath copy_miniapps_multidomain_test_data)


  add_mfem_miniapp(test-transfer-two-domains-overlapping-partition
    MAIN test-transfer-two-domains-overlapping-partition.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common mfem mfem-common-ecm2)

  add_mfem_miniapp(test-nonoverlapping-partition
    MAIN test-nonoverlapping-partition.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common mfem)

  add_mfem_miniapp(multidomain-two-domains-rf
    MAIN multidomain-two-domains-rf.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common mfem-common-ecm2 heat-ecm2 celldeath-ecm2 electrostatics_ecm2)  


  add_mfem_miniapp(multidomain-three-domains-celldeath-rf 
    MAIN multidomain-three-domains-celldeath-rf.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common mfem-common-ecm2 heat-ecm2 celldeath-ecm2 electrostatics_ecm2)
    

  add_mfem_miniapp(multidomain-three-domains-celldeath-rf-navier
    MAIN multidomain-three-domains-celldeath-rf-navier.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common mfem-common-ecm2 heat-ecm2 celldeath-ecm2 electrostatics_ecm2 navier-monolithic-ecm2)  

  add_mfem_miniapp(multidomain-one-domain-celldeath-rf-aniso
    MAIN multidomain-one-domain-celldeath-rf-aniso.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common mfem-common-ecm2 heat-ecm2 celldeath-ecm2 electrostatics_ecm2 )
      
endif()

