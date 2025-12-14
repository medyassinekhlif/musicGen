cmake_minimum_required(VERSION 3.15)

project(MidiRenderer VERSION 1.0 LANGUAGES CXX)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add JUCE
add_subdirectory(JUCE)

# Create executable
set(SOURCES
    src/main.cpp
    src/JuceBootstrap.cpp
    src/PluginPool.cpp
    src/RenderPipeline.cpp
    src/StreamingProcessor.cpp
    include/JuceBootstrap.h
    include/PluginPool.h
    include/RenderPipeline.h
    include/StreamingProcessor.h
)

# Add ServerApp only if Drogon is available
if(ENABLE_SERVER)
    list(APPEND SOURCES 
        src/ServerApp.cpp 
        include/ServerApp.h
        include/ChunkQueue.h
    )
endif()

add_executable(MidiRenderer ${SOURCES})

# Add include directory
target_include_directories(MidiRenderer PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Enable VST3 hosting
target_compile_definitions(MidiRenderer PRIVATE 
    JUCE_PLUGINHOST_VST3=1
    JUCE_PLUGINHOST_AU=0
    JUCE_PLUGINHOST_LADSPA=0
)

# Link JUCE modules
target_link_libraries(MidiRenderer
    PRIVATE
        juce::juce_audio_basics
        juce::juce_audio_formats
        juce::juce_audio_processors
        juce::juce_events
        juce::juce_core
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# Set output directory
set_target_properties(MidiRenderer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
)

# Copy executable to build root for easier access
add_custom_command(TARGET MidiRenderer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:MidiRenderer>
        ${CMAKE_BINARY_DIR}/$<CONFIG>/MidiRenderer.exe
)
