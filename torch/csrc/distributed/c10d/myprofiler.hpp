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

class ProfileTimer {
private:
    // This is the global list of pointers to every thread's local log
    static std::vector<std::vector<LogEntry>*>& get_all_thread_logs() {
        static std::vector<std::vector<LogEntry>*> all_logs;
        return all_logs;
    }
    
    static std::mutex& get_registry_mutex() {
        static std::mutex mtx;
        return mtx;
    }

public:
    static std::vector<LogEntry>& get_local_logs() {
        static thread_local std::vector<LogEntry> local_vec;
        static thread_local bool registered = false;
        
        if (!registered) {
            std::lock_guard<std::mutex> lock(get_registry_mutex());
            get_all_thread_logs().push_back(&local_vec);
            registered = true;
        }
        return local_vec;
    }

    explicit ProfileTimer(const char* func_name) : name(func_name) {
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        get_local_logs().push_back({std::this_thread::get_id(), name, "start", (long long)now});
    }

    ~ProfileTimer() {
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        get_local_logs().push_back({std::this_thread::get_id(), name, "end", (long long)now});
    }

    static void dump_to_file(const std::string& filename) {
        std::lock_guard<std::mutex> lock(get_registry_mutex());
        std::ofstream out(filename, std::ios::app);
        
        for (auto* thread_vec_ptr : get_all_thread_logs()) {
            for (const auto& entry : *thread_vec_ptr) {
                out << "bms#: " 
                    << entry.func_name << "," 
                    << entry.phase << "," 
                    << entry.timestamp << ","
                    << entry.tid << "\n";
            }
            // Optional: clear if you plan to keep running
            // thread_vec_ptr->clear(); 
        }
    }

private:
    const char* name;
};

#define MY_PROFILE() ProfileTimer timer_##__LINE__{__func__}
