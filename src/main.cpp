#include <iostream>
#include <memory>
#include <juce_core/juce_core.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include "JuceBootstrap.h"
#include "RenderPipeline.h"

int main(int argc, char* argv[])
{
    std::cout << "MIDI Renderer - Music Generation Tool" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Initialize JUCE - create bootstrap instance
    station::JuceBootstrap bootstrap;
    
    std::cout << "\nUsage:" << std::endl;
    std::cout << "  " << argv[0] << " <input.mid> <preset.vstpreset> <plugin.vst3> <output.wav>" << std::endl;
    std::cout << "  Use empty string \"\" for preset to use plugin defaults" << std::endl;
    std::cout << std::endl;
    
    if (argc >= 5)
    {
        juce::File midiFile(argv[1]);
        juce::String presetPath(argv[2]);
        juce::File pluginFile(argv[3]);
        juce::File outputFile(argv[4]);
        
        if (!midiFile.existsAsFile())
        {
            std::cerr << "Error: MIDI file not found: " << midiFile.getFullPathName() << std::endl;
            return 1;
        }
        
        // Preset is optional - use empty string to skip
        juce::File presetFile;
        if (!presetPath.isEmpty())
        {
            presetFile = juce::File(presetPath);
            if (!presetFile.existsAsFile())
            {
                std::cerr << "Warning: Preset file not found: " << presetFile.getFullPathName() << std::endl;
                std::cerr << "Continuing without preset (using plugin defaults)..." << std::endl;
                presetFile = juce::File();
            }
        }
        
        if (!pluginFile.exists())
        {
            std::cerr << "Error: Plugin not found: " << pluginFile.getFullPathName() << std::endl;
            return 1;
        }
        
        std::cout << "Processing MIDI file: " << midiFile.getFileName() << std::endl;
        if (presetFile != juce::File())
            std::cout << "Using preset: " << presetFile.getFileName() << std::endl;
        else
            std::cout << "Using plugin defaults (no preset)" << std::endl;
        std::cout << "Using plugin: " << pluginFile.getFileName() << std::endl;
        std::cout << "Output: " << outputFile.getFullPathName() << std::endl;
        
        // Create render request
        station::RenderRequest request;
        request.midiFile = midiFile;
        request.presetFile = presetFile;
        request.pluginPath = pluginFile.getFullPathName();
        request.wavOutputFile = outputFile;
        request.sampleRate = 48000.0;
        request.blockSize = 1024;
        
        // Create render pipeline
        station::RenderPipeline pipeline(request);
        
        // Process the file
        bool success = pipeline.renderToWav();
        
        if (success)
        {
            std::cout << "\nRendering completed successfully!" << std::endl;
            return 0;
        }
        else
        {
            std::cerr << "\nRendering failed!" << std::endl;
            return 1;
        }
    }
    else
    {
        std::cout << "No arguments provided. Please specify input files." << std::endl;
    }
    
    return 0;
}
