#pragma once

#include <memory>
#include <vector>
#include <atomic>

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_processors/juce_audio_processors.h>

#include "PluginPool.h"

namespace station
{

class StreamingProcessor
{
public:
    StreamingProcessor(PluginPool::InstancePtr pluginInstance,
                       double sampleRate = 48000.0,
                       int blockSizeIn = 512,
                       int bitDepthIn = 24);

    bool renderMidiToWav(const juce::File& midiFile,
                        const juce::File& presetFile,
                        const juce::File& outputWavFile);

    void stopProcessing();

private:
    bool loadPreset(juce::AudioPluginInstance& plugin, const juce::File& presetFile);
    juce::MidiFile readMidiFile(const juce::File& file, double sr, std::size_t& lengthOut);

    static std::size_t secondsToSamples(double seconds, double sampleRate);

private:
    PluginPool::InstancePtr pluginInstance_;
    double sampleRate;
    int blockSize;
    int bitDepth;
    std::atomic<bool> isProcessing{false};
};

} // namespace station
