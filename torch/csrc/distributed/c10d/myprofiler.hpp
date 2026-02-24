#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <mutex>
#include <thread>

struct LogEntry {
    std::thread::id tid; // Track which thread this came from
    const char* func_name;
    const char* phase;
    long long timestamp;
};

static std::vector<LogEntry> all_logs;
static std::mutex mtx;

class ProfileTimer {
private:
    // This is the global list of pointers to every thread's local log
    // static std::vector<std::vector<LogEntry>*>& get_all_thread_logs() {
    //     return all_logs;
    // }
    
    static std::mutex& get_registry_mutex() {
        return mtx;
    }

public:
    // static std::vector<LogEntry>& get_local_logs() {
    //     // static thread_local std::vector<LogEntry> local_vec;
    //     // static thread_local bool registered = false;
        
    //     // if (!registered) {
    //     //     std::lock_guard<std::mutex> lock(get_registry_mutex());
    //     //     get_all_thread_logs().push_back(&local_vec);
    //     //     registered = true;
    //     // }
    //     // return local_vec;
    // }

    explicit ProfileTimer(const char* func_name) : name(func_name) {
        auto now = std::chrono::steady_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
        ).count();
        all_logs.push_back({std::this_thread::get_id(), name, "start", (long long)ns});
    }

    ~ProfileTimer() {
        auto now = std::chrono::steady_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
        ).count();
        all_logs.push_back({std::this_thread::get_id(), name, "end", (long long)ns});
    }

    static void dump_to_file(const std::string& filename) {
        std::lock_guard<std::mutex> lock(get_registry_mutex());
        std::ofstream out(filename, std::ios::app);
        
        for (const auto& entry : all_logs) {
            out << "bms#: " 
                << entry.func_name << "," 
                << entry.phase << "," 
                << entry.timestamp << ","
                << entry.tid << "\n";
        }

        int count = all_logs.size();
        all_logs.clear();
        std::cout << "bms#: Dumped " << count << " log entries to " << filename << " and cleared all logs.\n";
    }

private:
    const char* name;
};

#define PROFILE_FUNCTION() ProfileTimer timer_##__LINE__{__func__}
