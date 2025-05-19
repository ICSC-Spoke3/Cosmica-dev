#include "EventSequence.hpp"

void EventSequence::AddEvent(const string &name) {
    GetCurrentSeq().events_.emplace_back(TimedEvent{name, Clock::now()});
}

EventSequence::EventSequencePtr EventSequence::StartSubsequence(const string &name, const bool parallel) {
    auto ss = std::make_shared<EventSequence>(name, parallel);
    GetCurrentSeq().events_.emplace_back(ss);
    if (!parallel_) current_subseq_.push(ss);
    return ss;
}

void EventSequence::StopSubsequence() {
    if (!parallel_) current_subseq_.pop();
}

std::vector<EventSequence::SummaryElement> EventSequence::Summary(const size_t maxDepth) const {
    std::vector<SummaryElement> summary;
    RecSummary(summary, *this, initial_event_, 0, maxDepth);
    return summary;
}

void EventSequence::Log(const spdlog::level::level_enum level, const size_t maxDepth) const {
    std::vector<SummaryElement> summary = Summary(maxDepth);
    for (size_t i = 0; i < summary.size(); ++i) {
        switch (const auto &[n, d, h, t, p, w] = summary[i]; t) {
            case SummaryElement::start:
                if (w) LogLine(level, h, FormatWrap(n, d, summary[++i].delta, p));
                else LogLine(level, h, FormatStart(n, d, p));
                break;
            case SummaryElement::end:
                LogLine(level, h, FormatEnd(d, p));
                break;
            default:
                LogLine(level, h, FormatEvent(n, d));
        }
    }
}


EventSequence::Duration EventSequence::TimeDiff(const TimePoint &t1, const TimePoint &t2) {
    return std::chrono::duration_cast<Duration>(t1 - t2);
}

// NOLINTNEXTLINE(*-no-recursion)
EventSequence::TimePoint EventSequence::LastEvent(const EventSequence &event) {
    if (event.parallel_) {
        TimePoint last = event.initial_event_, t;
        for (const auto &e: event.events_) {
            if ((std::holds_alternative<TimedEvent>(e) && (t = std::get<TimedEvent>(e).time) > last) ||
                (std::holds_alternative<EventSequencePtr>(e) && (t = LastEvent(*std::get<EventSequencePtr>(e))) > last))
                last = t;
        }
        return last;
    }

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
        out.emplace_back(
            event.sequence_name_, TimeDiff(last, prev), depth, SummaryElement::start, event.parallel_, true);
        last = LastEvent(event);
        out.emplace_back(event.sequence_name_, TimeDiff(last, prev), depth, SummaryElement::end, event.parallel_, true);
        return last;
    }

    out.emplace_back(event.sequence_name_, TimeDiff(event.initial_event_, prev), depth, SummaryElement::start,
                     event.parallel_);

    for (const auto &e: event.events_) {
        if (std::holds_alternative<TimedEvent>(e)) {
            auto [name, time] = std::get<TimedEvent>(e);
            out.emplace_back(name, TimeDiff(time, last), depth + 1);
            if (!event.parallel_ || time > last) last = time;
        } else {
            auto time = RecSummary(out, *std::get<EventSequencePtr>(e), event.parallel_ ? event.initial_event_ : last,
                              depth + 1, maxDepth);
            if (!event.parallel_ || time > last) last = time;
        }
    }

    out.emplace_back(event.sequence_name_, TimeDiff(last, prev), depth, SummaryElement::end, event.parallel_);

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

EventSequence::string EventSequence::FormatStart(const string &n, const Duration &d, bool parallel) {
    if (parallel) return n + FormatDelta(d) + " (parallel) {";
    return n + FormatDelta(d) + " [";
}

EventSequence::string EventSequence::FormatEnd(const Duration &d, const bool parallel) {
    if (parallel) return "} (" + FormatDuration(d) + ")";
    return "] (" + FormatDuration(d) + ")";
}

EventSequence::string EventSequence::FormatWrap(const string &n, const Duration &dd, const Duration &dt,
                                                const bool parallel) {
    return FormatStart(n, dd, parallel) + "..." + FormatEnd(dt, parallel);
}

EventSequence::string EventSequence::FormatEvent(const string &n, const Duration &d) {
    return n + FormatDelta(d);
}

void EventSequence::LogLine(const spdlog::level::level_enum level, const size_t pad, const string &line) {
    spdlog::log(level, "{:<{}}{}", "", pad * 2, line);
}
