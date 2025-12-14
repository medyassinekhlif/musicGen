#pragma once

#include <memory>
#include <string>
#include <vector>

#include "StreamingProcessor.h"

namespace station
{

struct RenderRequest
{
    juce::File midiFile;
    juce::File presetFile;
    juce::String pluginPath;
    juce::File wavOutputFile;
    double sampleRate{48000.0};
    int blockSize{1024};
};

class RenderPipeline
{
public:
    explicit RenderPipeline(RenderRequest request);
    ~RenderPipeline() = default;

    RenderPipeline(const RenderPipeline&) = delete;
    RenderPipeline& operator=(const RenderPipeline&) = delete;

    bool renderToWav();

private:
    RenderRequest request_;
};

using RenderPipelinePtr = std::shared_ptr<RenderPipeline>;

} // namespace station
