#include "EventSequence.hpp"

void EventSequence::AddEvent(const string &name) {
    GetCurrentSeq().events_.emplace_back(TimedEvent{name, Clock::now()});
}

void EventSequence::StartSubsequence(const string &name) {
    auto ss = std::make_shared<EventSequence>(name);
    GetCurrentSeq().events_.emplace_back(ss);
    current_subseq_.push(ss);
}

void EventSequence::StopSubsequence() {
    current_subseq_.pop();
}

std::vector<EventSequence::SummaryElement> EventSequence::Summary(const size_t maxDepth) const {
    std::vector<SummaryElement> summary;
    RecSummary(summary, *this, initial_event_, 0, maxDepth);
    return summary;
}

void EventSequence::Log(const spdlog::level::level_enum level, const size_t maxDepth) const {
    std::vector<SummaryElement> summary = Summary(maxDepth);
    for (size_t i = 0; i < summary.size(); ++i) {
        switch (const auto &[n, d, p, t, w] = summary[i]; t) {
            case SummaryElement::start:
                if (w) LogLine(level, p, FormatWrap(n, d, summary[++i].delta));
                else LogLine(level, p, FormatStart(n, d));
                break;
            case SummaryElement::end:
                LogLine(level, p, FormatEnd(d));
                break;
            default:
                LogLine(level, p, FormatEvent(n, d));
        }
    }
}


EventSequence::Duration EventSequence::TimeDiff(const TimePoint &t1, const TimePoint &t2) {
    return std::chrono::duration_cast<Duration>(t1 - t2);
}

// NOLINTNEXTLINE(*-no-recursion)
EventSequence::TimePoint EventSequence::LastEvent(const EventSequence &event) {
    if (event.events_.empty()) return event.initial_event_;
    const auto lastEvent = event.events_.back();
    if (std::holds_alternative<TimedEvent>(lastEvent)) {
        return std::get<TimedEvent>(lastEvent).time;
    }
    return LastEvent(*std::get<EventSequencePtr>(lastEvent));
}

// NOLINTNEXTLINE(*-no-recursion)
EventSequence::TimePoint EventSequence::RecSummary(std::vector<SummaryElement> &out, const EventSequence &event,
                                                   const TimePoint prev, const size_t depth, const size_t maxDepth) {
    TimePoint last = event.initial_event_;
    if (depth >= maxDepth) {
        out.emplace_back(event.sequence_name_, TimeDiff(last, prev), depth, SummaryElement::start, true);
        last = LastEvent(event);
        out.emplace_back(event.sequence_name_, TimeDiff(last, prev), depth, SummaryElement::end, true);
        return last;
    }

    out.emplace_back(event.sequence_name_, TimeDiff(event.initial_event_, prev), depth, SummaryElement::start);

    for (const auto &e: event.events_) {
        if (std::holds_alternative<TimedEvent>(e)) {
            auto [name, time] = std::get<TimedEvent>(e);
            out.emplace_back(name, TimeDiff(time, last), depth + 1);
            last = time;
        } else {
            last = RecSummary(out, *std::get<EventSequencePtr>(e), last, depth + 1, maxDepth);
        }
    }

    out.emplace_back(event.sequence_name_, TimeDiff(last, prev), depth, SummaryElement::end);

    return last;
}

template<typename Unit> requires std::chrono::__is_duration_v<Unit>
EventSequence::string EventSequence::FormatDuration(const Duration &d, string suffix) {
    auto v = std::chrono::duration_cast<Unit>(d).count();
    return (v == 0 ? "~" : "") + std::to_string(std::chrono::duration_cast<Unit>(d).count()) + suffix;
}

EventSequence::string EventSequence::FormatDuration(const Duration &d) {
    if (d > std::chrono::minutes(10)) return FormatDuration<std::chrono::minutes>(d, " m");
    if (d > std::chrono::seconds(10)) return FormatDuration<std::chrono::seconds>(d, " s");

    return FormatDuration<Duration>(d, " ms");
}

EventSequence::string EventSequence::FormatDelta(const Duration &d) {
    if (d.count() == 0) return "";
    return " (+" + FormatDuration(d) + ")";
}

EventSequence::string EventSequence::FormatStart(const string &n, const Duration &d) {
    return n + FormatDelta(d) + " {";
}

EventSequence::string EventSequence::FormatEnd(const Duration &d) {
    return "} (" + FormatDuration(d) + ")";
}

EventSequence::string EventSequence::FormatWrap(const string &n, const Duration &dd, const Duration &dt) {
    return FormatStart(n, dd) + "..." + FormatEnd(dt);
}

EventSequence::string EventSequence::FormatEvent(const string &n, const Duration &d) {
    return n + FormatDelta(d);
}

void EventSequence::LogLine(const spdlog::level::level_enum level, const size_t pad, const string &line) {
    spdlog::log(level, "{:<{}}{}", "", pad * 2, line);
}
