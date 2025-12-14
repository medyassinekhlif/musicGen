#include "StreamingProcessor.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace station
{
namespace
{
    constexpr double kDefaultPrerollSeconds = 0.5;
    constexpr double kTailSeconds = 6.0;
}

StreamingProcessor::StreamingProcessor(PluginPool::InstancePtr pluginInstance,
                                       double sampleRateIn,
                                       int blockSizeIn,
                                       int bitDepthIn)
    : pluginInstance_(std::move(pluginInstance)),
      sampleRate(sampleRateIn),
      blockSize(blockSizeIn),
      bitDepth(bitDepthIn),
      isProcessing(false)
{
}

std::size_t StreamingProcessor::secondsToSamples(double seconds, double sr)
{
    return static_cast<std::size_t>(std::round(seconds * sr));
}

bool StreamingProcessor::renderMidiToWav(const juce::File &midiFile,
                                         const juce::File &presetFile,
                                         const juce::File &outputWavFile)
{
    const auto tag = juce::String("[") + midiFile.getFileNameWithoutExtension() + juce::String("] ");

    if (!pluginInstance_)
    {
        std::cerr << tag << "Plugin instance unavailable" << std::endl;
        return false;
    }

    auto resetGuard = [&]() {
        if (pluginInstance_)
        {
            pluginInstance_->resetState();
        }
    };

    pluginInstance_->ensurePrepared(sampleRate, blockSize);
    auto &plugin = pluginInstance_->plugin();
    plugin.reset();
    plugin.suspendProcessing(false);
    plugin.setNonRealtime(true);

    juce::AudioPluginInstance::BusesLayout layout = plugin.getBusesLayout();
    if (layout.outputBuses.isEmpty())
    {
        layout.outputBuses.add(juce::AudioChannelSet::stereo());
    }
    else
    {
        layout.outputBuses.getReference(0) = juce::AudioChannelSet::stereo();
    }
    plugin.setBusesLayout(layout);

    const auto totalNumOutputChannels = static_cast<unsigned int>(std::max(1, plugin.getTotalNumOutputChannels()));

    if (presetFile.existsAsFile())
    {
        std::cout << tag << "Loading preset: " << presetFile.getFullPathName() << std::endl;
        if (!loadPreset(plugin, presetFile))
        {
            std::cerr << tag << "Failed to load preset" << std::endl;
        }
        
        // Re-prepare plugin after preset change to reinitialize audio buffers
        pluginInstance_->ensurePrepared(sampleRate, blockSize);
        plugin.reset();
        
        // Process a few silent blocks to let the plugin settle
        juce::AudioBuffer<float> settleBuffer(static_cast<int>(totalNumOutputChannels), blockSize);
        juce::MidiBuffer emptyMidi;
        for (int i = 0; i < 4; ++i)
        {
            settleBuffer.clear();
            plugin.processBlock(settleBuffer, emptyMidi);
        }
    }

    std::size_t midiLengthSamples = 0;
    auto midi = readMidiFile(midiFile, sampleRate, midiLengthSamples);
    if (midi.getNumTracks() == 0)
    {
        std::cerr << tag << "MIDI file contains no tracks" << std::endl;
        resetGuard();
        return false;
    }

    juce::MidiMessageSequence combined;
    for (int track = 0; track < midi.getNumTracks(); ++track)
    {
        if (auto *seq = midi.getTrack(track))
        {
            combined.addSequence(*seq, 0.0);
        }
    }
    combined.sort();

    const int latencySamples = plugin.getLatencySamples();
    const auto preRollSamples = std::max<std::size_t>(static_cast<std::size_t>(latencySamples),
                                                      secondsToSamples(kDefaultPrerollSeconds, sampleRate));
    const auto tailSamples = secondsToSamples(kTailSeconds, sampleRate);
    const std::size_t totalSamples = midiLengthSamples + tailSamples;

    // Create WAV writer
    outputWavFile.getParentDirectory().createDirectory();
    juce::WavAudioFormat wavFormat;
    auto wavStream = outputWavFile.createOutputStream();
    if (!wavStream)
    {
        std::cerr << tag << "Failed to open WAV output stream" << std::endl;
        resetGuard();
        return false;
    }

    std::unique_ptr<juce::AudioFormatWriter> wavWriter(
        wavFormat.createWriterFor(wavStream.release(),
                                 sampleRate,
                                 totalNumOutputChannels,
                                 32,
                                 {},
                                 0));
    if (!wavWriter)
    {
        std::cerr << tag << "Failed to create WAV writer" << std::endl;
        resetGuard();
        return false;
    }

    std::cout << tag << "Writing WAV to: " << outputWavFile.getFullPathName() << std::endl;

    juce::AudioBuffer<float> audioBuffer(static_cast<int>(totalNumOutputChannels), blockSize);
    juce::MidiBuffer midiBuffer;

    const int totalEvents = combined.getNumEvents();
    int eventIndex = 0;

    int64_t blockStartSample = -static_cast<int64_t>(preRollSamples);
    const int64_t finalSample = static_cast<int64_t>(totalSamples);

    isProcessing.store(true);
    
    while (blockStartSample < finalSample && isProcessing.load())
    {
        midiBuffer.clear();

        const int64_t blockEndSample = blockStartSample + blockSize;
        while (eventIndex < totalEvents)
        {
            const auto *holder = combined.getEventPointer(eventIndex);
            if (!holder)
            {
                break;
            }

            const auto eventSample = static_cast<int64_t>(std::llround(holder->message.getTimeStamp() * sampleRate)) -
                                     static_cast<int64_t>(preRollSamples);

            if (eventSample >= blockEndSample)
            {
                break;
            }

            if (eventSample >= blockStartSample)
            {
                const int offset = static_cast<int>(eventSample - blockStartSample);
                midiBuffer.addEvent(holder->message, offset);
            }
            ++eventIndex;
        }

        audioBuffer.clear();
        plugin.processBlock(audioBuffer, midiBuffer);

        // Write audio directly to WAV file (no chunking)
        if (blockStartSample >= 0)
        {
            const int64_t samplesRemaining = finalSample - blockStartSample;
            const int samplesToCopy = static_cast<int>(std::min<int64_t>(blockSize, samplesRemaining));
            
            if (samplesToCopy > 0)
            {
                wavWriter->writeFromAudioSampleBuffer(audioBuffer, 0, samplesToCopy);
            }
        }

        blockStartSample += blockSize;
    }

    wavWriter.reset();
    isProcessing.store(false);
    resetGuard();

    std::cout << tag << "Processing complete. WAV written to: " << outputWavFile.getFullPathName() << std::endl;
    return true;
}

void StreamingProcessor::stopProcessing()
{
    isProcessing.store(false);
    std::cout << "Stop signal received" << std::endl;
}

bool StreamingProcessor::loadPreset(juce::AudioPluginInstance &plugin, const juce::File &presetFile)
{
    if (!presetFile.existsAsFile())
    {
        return false;
    }

    juce::MemoryBlock presetData;
    if (!presetFile.loadFileAsData(presetData))
    {
        std::cerr << "Failed to read preset file: " << presetFile.getFullPathName() << std::endl;
        return false;
    }

    plugin.suspendProcessing(true);

    bool applied = false;

#if JUCE_PLUGINHOST_VST3
    if (presetFile.hasFileExtension("vstpreset"))
    {
        // Use JUCE's ExtensionsVisitor hook so we respect the VST3 preset container format.
        struct VST3PresetApplier : juce::ExtensionsVisitor
        {
            explicit VST3PresetApplier(juce::MemoryBlock& data) : presetData(data) {}
            
            bool success = false;

            void visitVST3Client(const ExtensionsVisitor::VST3Client& client) override
            {
                success = client.setPreset(juce::MemoryBlock(presetData.getData(), presetData.getSize()));
            }

            juce::MemoryBlock& presetData;
        };
        
        VST3PresetApplier visitor(presetData);
        plugin.getExtensions(visitor);
        applied = visitor.success;
    }
#endif

    if (!applied)
    {
        plugin.setStateInformation(presetData.getData(), static_cast<int>(presetData.getSize()));
        applied = true;
    }

    plugin.suspendProcessing(false);
    return applied;
}

juce::MidiFile StreamingProcessor::readMidiFile(const juce::File &file,
                                                double sr,
                                                std::size_t &lengthOut)
{
    juce::MidiFile midi;
    lengthOut = 0;

    auto stream = file.createInputStream();
    if (!stream)
    {
        std::cerr << "Unable to open MIDI file: " << file.getFullPathName() << std::endl;
        return midi;
    }

    if (!midi.readFrom(*stream, true))
    {
        std::cerr << "Failed to parse MIDI file: " << file.getFullPathName() << std::endl;
        return midi;
    }

    midi.convertTimestampTicksToSeconds();

    const double endTimeSeconds = midi.getLastTimestamp();
    lengthOut = secondsToSamples(endTimeSeconds, sr);

    return midi;
}

} // namespace station
