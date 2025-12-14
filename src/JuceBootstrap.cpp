#include "JuceBootstrap.h"

#include <chrono>
#include <iostream>
#include <thread>

namespace station
{

JuceBootstrap::JuceBootstrap()
{
    juce::initialiseJuce_GUI();

    messageThread = std::make_unique<std::thread>([] {
        std::cout << "JUCE message thread started" << std::endl;

        auto* mm = juce::MessageManager::getInstance();
        mm->setCurrentThreadAsMessageThread();

        class StopChecker : public juce::Timer
        {
        public:
            void timerCallback() override
            {
                if (shouldStop)
                {
                    juce::MessageManager::getInstance()->stopDispatchLoop();
                }
            }
        };

        StopChecker checker;
        checker.startTimer(100);

        std::cout << "Starting message dispatch loop..." << std::endl;
        mm->runDispatchLoop();
        std::cout << "JUCE message thread stopped" << std::endl;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::cout << "JUCE initialization complete" << std::endl;
}

JuceBootstrap::~JuceBootstrap()
{
    shouldStop = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    if (messageThread && messageThread->joinable())
    {
        messageThread->join();
    }

    juce::shutdownJuce_GUI();
}

} // namespace station
