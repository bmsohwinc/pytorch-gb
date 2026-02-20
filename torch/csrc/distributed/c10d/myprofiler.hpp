#include <iostream>
#include <chrono>
#include <string>

class ProfileTimer {
public:
    // We take the function name as a string to store it for the destructor
    explicit ProfileTimer(const char* func_name) : name(func_name) {
        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "bms#: " << name << ",start," 
                  << start.time_since_epoch().count() << std::endl;
    }

    ~ProfileTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "bms#: " << name << ",end," 
                  << end.time_since_epoch().count() << std::endl;
    }

private:
    const char* name;
};

#define PROFILE_FUNCTION() ProfileTimer timer_##__LINE__{__FUNCTION__}
