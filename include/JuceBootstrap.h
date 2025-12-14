#pragma once

#include <atomic>
#include <memory>
#include <thread>

#include <juce_gui_basics/juce_gui_basics.h>

namespace station
{

class JuceBootstrap
{
public:
    JuceBootstrap();
    ~JuceBootstrap();

    JuceBootstrap(const JuceBootstrap&) = delete;
    JuceBootstrap& operator=(const JuceBootstrap&) = delete;

private:
    static inline std::atomic<bool> shouldStop{false};
    std::unique_ptr<std::thread> messageThread;
};

} // namespace station
