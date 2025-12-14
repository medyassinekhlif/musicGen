#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <juce_audio_processors/juce_audio_processors.h>

namespace station
{

class PluginPool
{
public:
    class Instance;
    using InstancePtr = std::shared_ptr<Instance>;

    static PluginPool &get();

    void preload(const std::vector<juce::String> &pluginPaths,
                 int copiesPerPlugin = 1);

    InstancePtr acquire(const juce::String &pluginPath);

private:
    struct Entry;

    PluginPool();
    PluginPool(const PluginPool &) = delete;
    PluginPool &operator=(const PluginPool &) = delete;

    std::shared_ptr<Entry> getOrCreateEntry(const juce::String &pluginPath);
    std::unique_ptr<juce::AudioPluginInstance> createInstance(const juce::String &pluginPath);
    void releaseInstance(const std::shared_ptr<Entry> &entry, std::size_t index);

    juce::AudioPluginFormatManager formatManager_;
    std::mutex formatMutex_;
    std::mutex entriesMutex_;
    std::unordered_map<std::string, std::shared_ptr<Entry>> entries_;

public:
    class Instance
    {
    public:
        Instance(std::shared_ptr<Entry> entry,
                 std::size_t index,
                 PluginPool &owner);
        ~Instance();

        Instance(const Instance &) = delete;
        Instance &operator=(const Instance &) = delete;

        juce::AudioPluginInstance &plugin();
        void ensurePrepared(double sampleRate, int blockSize);
        void resetState();

    private:
        std::shared_ptr<Entry> entry_;
        std::size_t index_;
        PluginPool &owner_;
        bool released_{false};
    };
};

} // namespace station
