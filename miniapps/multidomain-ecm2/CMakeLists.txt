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
# Add subdirectories
add_subdirectory(dirichlet-neumann)
add_subdirectory(osm)
add_subdirectory(osm-2d)

