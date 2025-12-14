#include "RenderPipeline.h"

#include <iostream>

#include "PluginPool.h"

namespace station
{
    RenderPipeline::RenderPipeline(RenderRequest request)
        : request_(std::move(request))
    {
    }

    bool RenderPipeline::renderToWav()
    {
        const auto tag = juce::String("[") + request_.midiFile.getFileNameWithoutExtension() + juce::String("] ");

        if (!request_.midiFile.existsAsFile())
        {
            std::cerr << tag << "MIDI file does not exist: " << request_.midiFile.getFullPathName() << std::endl;
            return false;
        }

        if (request_.pluginPath.isEmpty())
        {
            std::cerr << tag << "No plugin path specified" << std::endl;
            return false;
        }

        std::cout << tag << "Loading plugin: " << request_.pluginPath << std::endl;
        auto pluginHandle = PluginPool::get().acquire(request_.pluginPath);
        if (!pluginHandle)
        {
            std::cerr << tag << "Failed to acquire plugin at " << request_.pluginPath << std::endl;
            return false;
        }

        StreamingProcessor processor(std::move(pluginHandle),
                                     request_.sampleRate,
                                     request_.blockSize,
                                     24);

        std::cout << tag << "Rendering MIDI to WAV..." << std::endl;
        const bool success = processor.renderMidiToWav(request_.midiFile,
                                                       request_.presetFile,
                                                       request_.wavOutputFile);

        if (success)
        {
            std::cout << tag << "==> Successfully rendered to: " << request_.wavOutputFile.getFullPathName() << std::endl;
        }
        else
        {
            std::cerr << tag << "==> Rendering failed" << std::endl;
        }

        return success;
    }

} // namespace station
