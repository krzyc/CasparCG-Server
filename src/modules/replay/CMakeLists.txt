cmake_minimum_required (VERSION 2.6)
project (replay)

if(MSVC)
    set(LIBJPEGTURBO_INCLUDE_PATH "${PACKAGES_FOLDER}/libjpeg-turbo/include/win32")
    set(LIBJPEGTURBO_BIN_PATH "${PACKAGES_FOLDER}/libjpeg-turbo/bin/win32")
    link_directories("${PACKAGES_FOLDER}/libjpeg-turbo/lib/win32")
	casparcg_add_runtime_dependency("${LIBJPEGTURBO_BIN_PATH}/jpeg62.dll")
else()
	casparcg_add_runtime_dependency("${LIBJPEGTURBO_BIN_PATH}/libjpeg.so.62")
endif()

set(SOURCES
		consumer/replay_consumer.cpp

		producer/replay_producer.cpp

		util/file_operations.cpp
		util/frame_operations.cpp

		replay.cpp
)
set(HEADERS
		consumer/replay_consumer.h

		producer/replay_producer.h

		util/file_operations.h
		util/frame_operations.h

		replay.h
)

add_library(replay ${SOURCES} ${HEADERS})

include_directories(..)
include_directories(../..)
include_directories(${LIBJPEGTURBO_INCLUDE_PATH})
include_directories(${BOOST_INCLUDE_PATH})
include_directories(${TBB_INCLUDE_PATH})
include_directories(${RXCPP_INCLUDE_PATH})
include_directories(${GLEW_INCLUDE_PATH})
include_directories(${ASMLIB_INCLUDE_PATH})

set_target_properties(replay PROPERTIES FOLDER modules)
source_group(sources ./*)
source_group(sources\\consumer consumer/*)
source_group(sources\\producer producer/*)
source_group(sources\\util util/*)

if(MSVC)
	target_link_libraries(replay
		common
		core

		"${PACKAGES_FOLDER}/libjpeg-turbo/lib/win32/jpeg.lib"
	)
else()
	target_link_libraries(replay
		common
		core

		jpeg
	)
endif()

casparcg_add_include_statement("modules/replay/replay.h")
casparcg_add_init_statement("replay::init" "replay")
casparcg_add_module_project("replay")
