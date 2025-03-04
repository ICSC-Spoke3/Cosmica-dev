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

    struct SummaryString {
        string summary;
        size_t depth;
    };

    struct SummaryElement {
        enum Type { event, start, end };

        string name;
        Duration delta;
        size_t depth;
        Type type = event;
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

        SummaryElement(string name, const Duration &delta, const size_t depth, const Type type, const bool warp)
            : name(std::move(name)),
              delta(delta),
              depth(depth),
              type(type),
              warp(warp) {
        }
    };

    explicit EventSequence(string name) : sequence_name_(std::move(name)), initial_event_(Clock::now()) {
    }

    void AddEvent(const string &name);

    void StartSubsequence(const string &name);

    void StopSubsequence();

    void StopStartSubsequence(const string &name) {
        StopSubsequence();
        StartSubsequence(name);
    }

    [[nodiscard]] std::vector<SummaryElement> Summary(size_t maxDepth = full_depth_) const;

    void Log(spdlog::level::level_enum level, size_t maxDepth = full_depth_) const;

private:
    string sequence_name_;
    TimePoint initial_event_;
    std::vector<std::variant<TimedEvent, EventSequencePtr> > events_;
    std::stack<EventSequencePtr> current_subseq_;

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

    static string FormatStart(const string &, const Duration &);

    static string FormatEnd(const Duration &);

    static string FormatWrap(const string &, const Duration &, const Duration &);

    static string FormatEvent(const string &, const Duration &);

    static void LogLine(spdlog::level::level_enum, size_t, const string &);
};

