#pragma once

#include <chrono>
#include <utility>
#include <vector>
#include <stack>

class EventSequence {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::milliseconds;
    using string = std::string;
    using EventSequencePtr = std::shared_ptr<EventSequence>;

    static constexpr size_t full_depth_ = std::numeric_limits<size_t>::max();

    struct TimedEvent {
        string name;
        TimePoint time;
    };

    struct SummaryElement {
        enum Type { event, start, end };

        string name;
        Duration delta;
        size_t depth;
        Type type = event;
        bool parallel = false;
        bool warp = false;

        SummaryElement(string name, const Duration &delta, const size_t depth)
            : name(std::move(name)),
              delta(delta),
              depth(depth) {
        }

        SummaryElement(string name, const Duration &delta, const size_t depth, const Type type)
            : name(std::move(name)),
              delta(delta),
              depth(depth),
              type(type) {
        }

        SummaryElement(string name, const Duration &delta, const size_t depth, const Type type, const bool parallel)
            : name(std::move(name)),
              delta(delta),
              depth(depth),
              type(type),
              parallel(parallel) {
        }

        SummaryElement(string name, const Duration &delta, size_t depth, Type type, bool parallel, bool warp)
            : name(std::move(name)),
              delta(delta),
              depth(depth),
              type(type),
              parallel(parallel),
              warp(warp) {
        }
    };

    explicit EventSequence(string name) : sequence_name_(std::move(name)), initial_event_(Clock::now()) {
    }

    EventSequence(string name, const bool parallel) : sequence_name_(std::move(name)), initial_event_(Clock::now()),
                                                      parallel_(parallel) {
    }

    void AddEvent(const string &name);

    EventSequencePtr StartSubsequence(const string &name, bool parallel = false);

    void StopSubsequence();

    [[nodiscard]] std::vector<SummaryElement> Summary(size_t maxDepth = full_depth_) const;

    void Log(spdlog::level::level_enum level, size_t maxDepth = full_depth_) const;

private:
    string sequence_name_;
    TimePoint initial_event_;
    std::vector<std::variant<TimedEvent, EventSequencePtr> > events_;
    std::stack<EventSequencePtr> current_subseq_;
    bool parallel_ = false;

    EventSequence &GetCurrentSeq() {
        return current_subseq_.empty() ? *this : *current_subseq_.top();
    }

    static Duration TimeDiff(const TimePoint &, const TimePoint &);

    static TimePoint LastEvent(const EventSequence &);

    static TimePoint RecSummary(std::vector<SummaryElement> &, const EventSequence &, TimePoint, size_t, size_t);

    template<typename Unit> requires std::chrono::__is_duration_v<Unit>
    static string FormatDuration(const Duration &, string suffix);

    static string FormatDuration(const Duration &);

    static string FormatDelta(const Duration &);

    static string FormatStart(const string &, const Duration &, bool);

    static string FormatEnd(const Duration &, bool);

    static string FormatWrap(const string &, const Duration &, const Duration &, bool);

    static string FormatEvent(const string &, const Duration &);

    static void LogLine(spdlog::level::level_enum, size_t, const string &);
};
