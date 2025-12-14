#include "PluginPool.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include <juce_audio_basics/juce_audio_basics.h>

namespace station
{
namespace
{
    constexpr int kDefaultCopiesPerPlugin = 2;
}

struct PluginPool::Entry
{
    juce::String path;
    std::vector<std::unique_ptr<juce::AudioPluginInstance>> instances;
    std::vector<bool> inUse;
    std::vector<double> preparedSampleRates;
    std::vector<int> preparedBlockSizes;
    std::mutex mutex;
};

PluginPool::PluginPool()
{
    formatManager_.addDefaultFormats();
}

PluginPool &PluginPool::get()
{
    static PluginPool pool;
    return pool;
}

void PluginPool::preload(const std::vector<juce::String> &pluginPaths,
                         int copiesPerPlugin)
{
    const int copies = std::max(1, copiesPerPlugin);
    for (const auto &path : pluginPaths)
    {
        if (path.isEmpty())
        {
            continue;
        }

        auto entry = getOrCreateEntry(path);
        if (!entry)
        {
            continue;
        }

        std::lock_guard<std::mutex> lock(entry->mutex);
        while (static_cast<int>(entry->instances.size()) < copies)
        {
            auto instance = createInstance(path);
            if (!instance)
            {
                std::cerr << "PluginPool: failed to instantiate plugin at " << path << std::endl;
                break;
            }
            entry->preparedSampleRates.push_back(0.0);
            entry->preparedBlockSizes.push_back(0);
            entry->inUse.push_back(false);
            entry->instances.push_back(std::move(instance));
        }
    }
}

PluginPool::InstancePtr PluginPool::acquire(const juce::String &pluginPath)
{
    if (pluginPath.isEmpty())
    {
        return nullptr;
    }

    auto entry = getOrCreateEntry(pluginPath);
    if (!entry)
    {
        std::cerr << "PluginPool: unable to load plugin at " << pluginPath << std::endl;
        return nullptr;
    }

    std::unique_lock<std::mutex> lock(entry->mutex);

    auto findFreeIndex = [&]() -> int {
        for (std::size_t i = 0; i < entry->instances.size(); ++i)
        {
            if (!entry->inUse[i])
            {
                return static_cast<int>(i);
            }
        }
        return -1;
    };

    int freeIndex = findFreeIndex();
    if (freeIndex < 0)
    {
        auto instance = createInstance(pluginPath);
        if (!instance)
        {
            std::cerr << "PluginPool: unable to expand pool for " << pluginPath << std::endl;
            return nullptr;
        }
        entry->preparedSampleRates.push_back(0.0);
        entry->preparedBlockSizes.push_back(0);
        entry->inUse.push_back(false);
        entry->instances.push_back(std::move(instance));
        freeIndex = static_cast<int>(entry->instances.size() - 1);
    }

    entry->inUse[freeIndex] = true;
    auto handle = std::make_shared<Instance>(entry, static_cast<std::size_t>(freeIndex), *this);
    return handle;
}

std::shared_ptr<PluginPool::Entry> PluginPool::getOrCreateEntry(const juce::String &pluginPath)
{
    std::shared_ptr<Entry> entry;
    {
        std::lock_guard<std::mutex> guard(entriesMutex_);
        auto key = pluginPath.toStdString();
        auto it = entries_.find(key);
        if (it != entries_.end())
        {
            entry = it->second;
        }
        else
        {
            entry = std::make_shared<Entry>();
            entry->path = pluginPath;
            entries_.emplace(key, entry);
        }
    }
    return entry;
}

std::unique_ptr<juce::AudioPluginInstance> PluginPool::createInstance(const juce::String &pluginPath)
{
    std::lock_guard<std::mutex> formatLock(formatMutex_);
    juce::OwnedArray<juce::PluginDescription> descriptions;
    juce::PluginDescription description;

    for (int i = 0; i < formatManager_.getNumFormats(); ++i)
    {
        auto *format = formatManager_.getFormat(i);
        if (!format)
        {
            continue;
        }

        descriptions.clear();
        format->findAllTypesForFile(descriptions, pluginPath);
        if (descriptions.isEmpty())
        {
            continue;
        }

        description = *descriptions.getFirst();
        juce::String error;
        auto instance = format->createInstanceFromDescription(description, 48000.0, 512, error);
        if (!instance)
        {
            std::cerr << "PluginPool: createInstanceFromDescription failed: " << error << std::endl;
            continue;
        }

        std::cout << "PluginPool: Loaded plugin " << instance->getName() << " from " << pluginPath << std::endl;
        return instance;
    }

    std::cerr << "PluginPool: no plugin formats matched " << pluginPath << std::endl;
    return nullptr;
}

void PluginPool::releaseInstance(const std::shared_ptr<Entry> &entry, std::size_t index)
{
    if (!entry)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(entry->mutex);
    if (index >= entry->instances.size())
    {
        return;
    }

    auto &plugin = entry->instances[index];
    entry->inUse[index] = false;
}

PluginPool::Instance::Instance(std::shared_ptr<Entry> entry,
                               std::size_t index,
                               PluginPool &owner)
    : entry_(std::move(entry)),
      index_(index),
      owner_(owner)
{
}

PluginPool::Instance::~Instance()
{
    if (!released_)
    {
        owner_.releaseInstance(entry_, index_);
        released_ = true;
    }
}

juce::AudioPluginInstance &PluginPool::Instance::plugin()
{
    return *entry_->instances[index_];
}

void PluginPool::Instance::ensurePrepared(double sampleRate, int blockSize)
{
    std::lock_guard<std::mutex> lock(entry_->mutex);
    if (index_ >= entry_->instances.size())
    {
        return;
    }

    auto &pluginInstance = entry_->instances[index_];
    auto &preparedRate = entry_->preparedSampleRates[index_];
    auto &preparedBlock = entry_->preparedBlockSizes[index_];

    const bool needsPrepare = std::abs(preparedRate - sampleRate) > 1e-3 || preparedBlock != blockSize;
    if (needsPrepare)
    {
        pluginInstance->releaseResources();
        pluginInstance->prepareToPlay(sampleRate, blockSize);
        preparedRate = sampleRate;
        preparedBlock = blockSize;
    }
    pluginInstance->setNonRealtime(true);
}

void PluginPool::Instance::resetState()
{
    std::lock_guard<std::mutex> lock(entry_->mutex);
    if (index_ >= entry_->instances.size())
    {
        return;
    }

    auto &pluginInstance = entry_->instances[index_];
    pluginInstance->reset();
    if (entry_->preparedSampleRates[index_] > 0.0)
    {
        juce::MidiBuffer allNotesOff;
        for (int channel = 1; channel <= 16; ++channel)
        {
            allNotesOff.addEvent(juce::MidiMessage::allNotesOff(channel), 0);
            allNotesOff.addEvent(juce::MidiMessage::allSoundOff(channel), 0);
        }
        juce::AudioBuffer<float> scratch(pluginInstance->getTotalNumOutputChannels(), 32);
        scratch.clear();
        pluginInstance->processBlock(scratch, allNotesOff);
    }
}

} // namespace station
